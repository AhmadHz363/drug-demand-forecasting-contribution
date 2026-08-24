"""SHIELD-XR daily hurdle model, baselines, and class-conditional ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from app.forecasting.shield_xr.features import CAT_COLS, FEATURES_X, feature_row_to_frame
from app.forecasting.shield_xr.metrics import wape

RANDOM_STATE = 42


def _xy(frame: pd.DataFrame, ycol: str, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return frame[cols].copy(), frame[ycol].astype(float).values


@dataclass
class ShieldXRStackResult:
    pred: np.ndarray
    pred_valid: np.ndarray
    p_occur: np.ndarray
    p_spike: np.ndarray
    gate_config: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)
    isotonic_calibrators: dict[str, IsotonicRegression] = field(default_factory=dict)
    evt_frac: float = 0.0


@dataclass
class DailyEnsembleArtifacts:
    shield_xr: ShieldXRStackResult
    plain_tweedie: Any
    plain_l1: Any
    plain_pred_valid: np.ndarray
    plain_pred: np.ndarray
    plain_l1_pred_valid: np.ndarray
    plain_l1_pred: np.ndarray
    class_winner: dict[str, str]
    class_bias: dict[str, float]
    feature_cols: list[str]
    code2id: dict[str, int]
    cat2id: dict[str, int]
    residual_q90: float = 0.0


def train_shieldxr_stack(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> ShieldXRStackResult:
    cats = [c for c in CAT_COLS if c in feature_cols]
    x_train, y_train = _xy(train, "occur", feature_cols)
    x_valid, y_valid = _xy(valid, "occur", feature_cols)
    x_test, _ = _xy(test, "occur", feature_cols)

    occur = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=47,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    occur.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    p_occ_valid = occur.predict_proba(x_valid)[:, 1]
    p_occ_test = occur.predict_proba(x_test)[:, 1]

    isotonic: dict[str, IsotonicRegression] = {}
    p_occ_valid_cal = p_occ_valid.copy()
    p_occ_test_cal = p_occ_test.copy()
    for cls in valid["sb_class"].unique():
        mask_valid = (valid["sb_class"] == cls).values
        mask_test = (test["sb_class"] == cls).values
        if mask_valid.sum() < 30 or valid.loc[mask_valid, "occur"].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_occ_valid[mask_valid], valid.loc[mask_valid, "occur"].values)
        isotonic[str(cls)] = iso
        p_occ_valid_cal[mask_valid] = iso.transform(p_occ_valid[mask_valid])
        if mask_test.sum():
            p_occ_test_cal[mask_test] = iso.transform(p_occ_test[mask_test])
    p_occ_valid, p_occ_test = p_occ_valid_cal, p_occ_test_cal

    mask_train_normal = (train["demand"] > 0) & (train["spike"] == 0)
    mask_valid_normal = (valid["demand"] > 0) & (valid["spike"] == 0)
    normal = lgb.LGBMRegressor(
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="tweedie",
        tweedie_variance_power=1.15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    normal.fit(
        *_xy(train.loc[mask_train_normal], "demand", feature_cols),
        sample_weight=train.loc[mask_train_normal, "_w"] if "_w" in train.columns else None,
        eval_set=[_xy(valid.loc[mask_valid_normal], "demand", feature_cols)] if mask_valid_normal.sum() else None,
        eval_sample_weight=[valid.loc[mask_valid_normal, "_w"]] if (mask_valid_normal.sum() and "_w" in valid.columns) else None,
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(50, verbose=False)] if mask_valid_normal.sum() else None,
    )
    yhat_n_valid = np.clip(normal.predict(_xy(valid, "demand", feature_cols)[0]), 0, None)
    yhat_n_test = np.clip(normal.predict(_xy(test, "demand", feature_cols)[0]), 0, None)

    full_reg = lgb.LGBMRegressor(
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="tweedie",
        tweedie_variance_power=1.15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    full_reg.fit(
        *_xy(train, "demand", feature_cols),
        sample_weight=train["_w"] if "_w" in train.columns else None,
        eval_set=[_xy(valid, "demand", feature_cols)],
        eval_sample_weight=[valid["_w"]] if "_w" in valid.columns else None,
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    yhat_full_valid = np.clip(full_reg.predict(_xy(valid, "demand", feature_cols)[0]), 0, None)
    yhat_full_test = np.clip(full_reg.predict(_xy(test, "demand", feature_cols)[0]), 0, None)

    spike_clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=47,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    spike_clf.fit(
        *_xy(train, "spike", feature_cols),
        eval_set=[_xy(valid, "spike", feature_cols)],
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    p_spk_valid = spike_clf.predict_proba(_xy(valid, "spike", feature_cols)[0])[:, 1]
    p_spk_test = spike_clf.predict_proba(_xy(test, "spike", feature_cols)[0])[:, 1]

    mask_train_spike = train["spike"] == 1
    if mask_train_spike.sum() >= 40:
        spike_q = lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=31,
            objective="quantile",
            alpha=0.90,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        mask_valid_spike = valid["spike"] == 1
        spike_q.fit(
            *_xy(train.loc[mask_train_spike], "demand", feature_cols),
            eval_set=[_xy(valid.loc[mask_valid_spike], "demand", feature_cols)] if mask_valid_spike.sum() else None,
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(40, verbose=False)] if mask_valid_spike.sum() else None,
        )
        yhat_s_valid = np.clip(spike_q.predict(_xy(valid, "demand", feature_cols)[0]), 0, None)
        yhat_s_test = np.clip(spike_q.predict(_xy(test, "demand", feature_cols)[0]), 0, None)
    else:
        p90 = train.groupby("CODE")["demand"].quantile(0.90)
        yhat_s_valid = valid["CODE"].map(p90).fillna(train["demand"].quantile(0.90)).values
        yhat_s_test = test["CODE"].map(p90).fillna(train["demand"].quantile(0.90)).values
        spike_q = None

    resid = valid["demand"].values - yhat_n_valid
    threshold = np.quantile(resid, 0.90)
    excess = resid[resid > threshold] - threshold
    if len(excess) >= 30:
        shape, _, scale = genpareto.fit(excess, floc=0)
        evt_abs = float(scale / (1 - shape)) if shape < 1 else float(np.mean(excess))
    else:
        evt_abs = float(np.mean(excess)) if len(excess) else 0.0
    evt_abs = max(evt_abs, 0.0)
    baseline_scale = float(np.mean(yhat_n_valid[yhat_n_valid > 0])) if (yhat_n_valid > 0).any() else 1.0
    evt_frac = float(np.clip(evt_abs / max(baseline_scale, 1e-6), 0.0, 3.0))
    yhat_s_valid = np.maximum(yhat_s_valid, yhat_n_valid * (1 + evt_frac))
    yhat_s_test = np.maximum(yhat_s_test, yhat_n_test * (1 + evt_frac))

    is_smoothish_valid = valid["sb_class"].isin(["smooth", "erratic"]).values
    is_smoothish_test = test["sb_class"].isin(["smooth", "erratic"]).values
    best = {"wape": 1e9, "t_spk": 1.01, "hard_occ_t": None, "w_smooth": 0.0, "bias_c": 1.0}
    for t_spk in (1.01, 0.85, 0.70, 0.50, 0.35, 0.20):
        use_spk_valid = p_spk_valid >= t_spk
        mag_valid = np.where(use_spk_valid, yhat_s_valid, yhat_n_valid)
        for hard_occ_t in (None, 0.50, 0.40, 0.30, 0.20, 0.10):
            occ_term_valid = p_occ_valid if hard_occ_t is None else (p_occ_valid >= hard_occ_t).astype(float)
            hurdle_valid = occ_term_valid * mag_valid
            for w_smooth in np.linspace(0.0, 0.9, 10):
                pred_valid = hurdle_valid.copy()
                pred_valid[is_smoothish_valid] = (
                    (1 - w_smooth) * hurdle_valid[is_smoothish_valid]
                    + w_smooth * yhat_full_valid[is_smoothish_valid]
                )
                s_pred = pred_valid.sum()
                bias_c = float(np.clip(valid["demand"].sum() / s_pred, 0.5, 2.0)) if s_pred > 0 else 1.0
                score = wape(valid["demand"].values, pred_valid * bias_c)
                if score < best["wape"]:
                    best = {
                        "wape": score,
                        "t_spk": t_spk,
                        "hard_occ_t": hard_occ_t,
                        "w_smooth": float(w_smooth),
                        "bias_c": bias_c,
                    }

    use_spk_test = p_spk_test >= best["t_spk"]
    mag_test = np.where(use_spk_test, yhat_s_test, yhat_n_test)
    occ_term_test = p_occ_test if best["hard_occ_t"] is None else (p_occ_test >= best["hard_occ_t"]).astype(float)
    hurdle_test = occ_term_test * mag_test
    pred_test = hurdle_test.copy()
    pred_test[is_smoothish_test] = (
        (1 - best["w_smooth"]) * hurdle_test[is_smoothish_test]
        + best["w_smooth"] * yhat_full_test[is_smoothish_test]
    )
    pred_test = np.clip(pred_test * best["bias_c"], 0, None)

    use_spk_valid_f = p_spk_valid >= best["t_spk"]
    mag_valid_f = np.where(use_spk_valid_f, yhat_s_valid, yhat_n_valid)
    occ_term_valid_f = p_occ_valid if best["hard_occ_t"] is None else (p_occ_valid >= best["hard_occ_t"]).astype(float)
    hurdle_valid_f = occ_term_valid_f * mag_valid_f
    pred_valid_f = hurdle_valid_f.copy()
    pred_valid_f[is_smoothish_valid] = (
        (1 - best["w_smooth"]) * hurdle_valid_f[is_smoothish_valid]
        + best["w_smooth"] * yhat_full_valid[is_smoothish_valid]
    )
    pred_valid_f = np.clip(pred_valid_f * best["bias_c"], 0, None)

    try:
        auc = roc_auc_score(valid["occur"], p_occ_valid)
    except ValueError:
        auc = float("nan")

    return ShieldXRStackResult(
        pred=pred_test,
        pred_valid=pred_valid_f,
        p_occur=p_occ_test,
        p_spike=p_spk_test,
        gate_config=best,
        evt_frac=evt_frac,
        isotonic_calibrators=isotonic,
        models={
            "occur": occur,
            "normal": normal,
            "full_reg": full_reg,
            "spike_clf": spike_clf,
            "spike_q": spike_q,
            "yhat_n": normal,
            "yhat_full": full_reg,
            "yhat_s_fallback_p90": train.groupby("CODE")["demand"].quantile(0.90).to_dict(),
        },
    )


def train_plain_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cats = [c for c in CAT_COLS if c in feature_cols]
    plain = lgb.LGBMRegressor(
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="tweedie",
        tweedie_variance_power=1.2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    plain.fit(
        *_xy(train, "demand", feature_cols),
        sample_weight=train["_w"],
        eval_set=[_xy(valid, "demand", feature_cols)],
        eval_sample_weight=[valid["_w"]],
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    plain_l1 = lgb.LGBMRegressor(
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    plain_l1.fit(
        *_xy(train, "demand", feature_cols),
        sample_weight=train["_w"],
        eval_set=[_xy(valid, "demand", feature_cols)],
        eval_sample_weight=[valid["_w"]],
        categorical_feature=cats,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    x_test = _xy(test, "demand", feature_cols)[0]
    x_valid = _xy(valid, "demand", feature_cols)[0]
    plain_pred = np.clip(plain.predict(x_test), 0, None)
    plain_pred_valid = np.clip(plain.predict(x_valid), 0, None)
    plain_l1_pred = np.clip(plain_l1.predict(x_test), 0, None)
    plain_l1_pred_valid = np.clip(plain_l1.predict(x_valid), 0, None)
    return plain, plain_l1, plain_pred, plain_pred_valid, plain_l1_pred, plain_l1_pred_valid


def select_class_ensemble(
    valid: pd.DataFrame,
    test: pd.DataFrame,
    shield_xr: ShieldXRStackResult,
    plain_pred_valid: np.ndarray,
    plain_pred: np.ndarray,
    plain_l1_pred_valid: np.ndarray,
    plain_l1_pred: np.ndarray,
) -> tuple[dict[str, str], dict[str, float], np.ndarray]:
    candidates = {
        "SHIELD-XR": (shield_xr.pred_valid, shield_xr.pred),
        "Plain": (plain_pred_valid, plain_pred),
        "Plain-L1": (plain_l1_pred_valid, plain_l1_pred),
    }
    class_winner: dict[str, str] = {}
    class_bias: dict[str, float] = {}
    ensemble_pred = np.zeros(len(test))
    for cls in valid["sb_class"].unique():
        mask = (valid["sb_class"] == cls).values
        if mask.sum() == 0:
            continue
        y_valid = valid.loc[mask, "demand"].values
        scores = {name: wape(y_valid, pred[mask]) for name, (pred, _) in candidates.items()}
        winner = min(scores, key=scores.get)
        class_winner[str(cls)] = winner
        va_pred = candidates[winner][0][mask]
        total = va_pred.sum()
        bias = float(np.clip(y_valid.sum() / total, 0.75, 1.35)) if total > 0 else 1.0
        class_bias[str(cls)] = bias
        mask_test = (test["sb_class"] == cls).values
        ensemble_pred[mask_test] = candidates[winner][1][mask_test] * bias
    ensemble_pred = np.clip(ensemble_pred, 0, None)
    return class_winner, class_bias, ensemble_pred


def predict_shieldxr_row(
    row: pd.Series,
    artifacts: DailyEnsembleArtifacts,
) -> float:
    """Predict a single row using the stored SHIELD-XR stack (inference helper)."""
    feature_cols = artifacts.feature_cols
    x = feature_row_to_frame(row, feature_cols)
    stack = artifacts.shield_xr
    gate = stack.gate_config
    cls = str(row.get("sb_class", "lumpy"))
    occur_model = stack.models["occur"]
    p_occ = float(occur_model.predict_proba(x)[0, 1])
    if cls in stack.isotonic_calibrators:
        p_occ = float(stack.isotonic_calibrators[cls].transform([p_occ])[0])

    normal = stack.models["normal"]
    full_reg = stack.models["full_reg"]
    spike_clf = stack.models["spike_clf"]
    yhat_n = max(0.0, float(normal.predict(x)[0]))
    yhat_full = max(0.0, float(full_reg.predict(x)[0]))
    p_spk = float(spike_clf.predict_proba(x)[0, 1])

    if p_spk >= gate["t_spk"]:
        spike_q = stack.models.get("spike_q")
        if spike_q is not None:
            yhat_s = max(0.0, float(spike_q.predict(x)[0]))
        else:
            fallback = stack.models.get("yhat_s_fallback_p90", {})
            yhat_s = float(fallback.get(str(row["CODE"]), yhat_n))
        yhat_s = max(yhat_s, yhat_n * (1 + stack.evt_frac))
        mag = yhat_s
    else:
        mag = yhat_n

    hard_occ = gate["hard_occ_t"]
    occ_term = p_occ if hard_occ is None else float(p_occ >= hard_occ)
    hurdle = occ_term * mag
    if cls in {"smooth", "erratic"}:
        w_smooth = gate["w_smooth"]
        pred = (1 - w_smooth) * hurdle + w_smooth * yhat_full
    else:
        pred = hurdle
    return max(0.0, pred * gate["bias_c"])


def predict_ensemble_row(row: pd.Series, artifacts: DailyEnsembleArtifacts) -> float:
    cls = str(row.get("sb_class", "lumpy"))
    winner = artifacts.class_winner.get(cls, "SHIELD-XR")
    bias = artifacts.class_bias.get(cls, 1.0)
    x = feature_row_to_frame(row, artifacts.feature_cols)
    if winner == "Plain":
        pred = max(0.0, float(artifacts.plain_tweedie.predict(x)[0]))
    elif winner == "Plain-L1":
        pred = max(0.0, float(artifacts.plain_l1.predict(x)[0]))
    else:
        pred = predict_shieldxr_row(row, artifacts)
    return max(0.0, pred * bias)
