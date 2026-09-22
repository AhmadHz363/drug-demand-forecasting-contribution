"""Chapter 5 cold-start baselines: ARIMA-on-analog and DDPFF-style.

Taken from the leave-drugs-out notebook that produced Table 5.3
(``CAMEO_RealData_ColdStart.ipynb``). The in-app hold-out scorer still
compares CAMEO only with the Analogous baseline in ``validation.py``.

The reported run used ``topk=5``, a 20-week horizon, and Syntetos–Boylan
labels as the DDPFF cluster labels. ARIMA is order (1, 0, 1), fit on the
single nearest analog. DDPFF-style blends an XGBoost cluster level and a
distance-weighted analog level with alpha 0.6, then applies an ARIMA
residual correction and a 7-point SPC level reset.
"""

from __future__ import annotations

import numpy as np

ARIMA_ORDER = (1, 0, 1)
DDPFF_ALPHA = 0.6
DDPFF_LEVEL_WEEKS = 8
DDPFF_RESIDUAL_WINDOW = 8
DDPFF_SPC_RUN = 7


def nearest_analog_series(
    new_attr_raw: np.ndarray,
    hist_attrs_raw: np.ndarray,
    hist_series: list[np.ndarray],
) -> np.ndarray:
    """Return the history of the library drug closest in raw attribute space."""
    distances = np.linalg.norm(hist_attrs_raw - new_attr_raw, axis=1)
    return np.asarray(hist_series[int(np.argmin(distances))], dtype=float)


def baseline_arima(analog_series: np.ndarray, horizon_weeks: int) -> np.ndarray:
    """Forecast ``horizon_weeks`` with ARIMA(1, 0, 1) fit on one analog series.

    Falls back to the analog mean if the fit fails.
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = np.asarray(analog_series, dtype=float)
    try:
        model = ARIMA(series, order=ARIMA_ORDER).fit()
        return np.asarray(model.forecast(steps=horizon_weeks), dtype=float)
    except Exception:
        return np.full(horizon_weeks, float(series.mean()))


def baseline_ddpff_style(
    new_attr_raw: np.ndarray,
    hist_attrs_raw: np.ndarray,
    cluster_labels: list[str],
    hist_series: list[np.ndarray],
    actual_demand: np.ndarray,
    topk: int = 3,
) -> np.ndarray:
    """DDPFF-style cluster, analog blend, residual ARIMA, and 7-point SPC.

    ``actual_demand`` is the new drug's observed launch weeks. The residual
    correction and the SPC reset are online: week ``t`` may use demand
    through ``t`` when updating the forecast for week ``t + 1``.

    The Chapter 5 run passed ``topk=5``. The function default of 3 matches
    the notebook definition.
    """
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.tsa.arima.model import ARIMA
    from xgboost import XGBClassifier

    label_to_id = {label: index for index, label in enumerate(sorted(set(cluster_labels)))}
    class_ids = np.array([label_to_id[label] for label in cluster_labels])
    classifier = XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        eval_metric="mlogloss",
    )
    classifier.fit(hist_attrs_raw, class_ids)
    predicted_cluster = classifier.predict(new_attr_raw.reshape(1, -1))[0]
    in_cluster = np.where(class_ids == predicted_cluster)[0]
    if len(in_cluster) < 2:
        in_cluster = np.arange(len(hist_attrs_raw))

    level_targets = np.array(
        [np.asarray(hist_series[index][:DDPFF_LEVEL_WEEKS], dtype=float).mean() for index in in_cluster]
    )
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=2,
        random_state=0,
    )
    regressor.fit(hist_attrs_raw[in_cluster], level_targets)
    cluster_level = float(regressor.predict(new_attr_raw.reshape(1, -1))[0])

    distances = np.linalg.norm(hist_attrs_raw - new_attr_raw, axis=1)
    order = np.argsort(distances)[:topk]
    weights = 1.0 / (distances[order] + 1e-3)
    weights = weights / weights.sum()
    analog_level = float(
        np.dot(
            weights,
            [np.asarray(hist_series[index][:DDPFF_LEVEL_WEEKS], dtype=float).mean() for index in order],
        )
    )
    init_level = DDPFF_ALPHA * cluster_level + (1.0 - DDPFF_ALPHA) * analog_level

    horizon = len(actual_demand)
    base_forecast = np.full(horizon, init_level)
    forecasts = base_forecast.copy()
    residuals: list[float] = []
    observed = np.asarray(actual_demand, dtype=float)
    for week in range(horizon):
        residuals.append(float(observed[week] - forecasts[week]))
        if len(residuals) >= 4:
            residual_window = np.array(residuals[-DDPFF_RESIDUAL_WINDOW:])
            try:
                next_error = float(ARIMA(residual_window, order=ARIMA_ORDER).fit().forecast(steps=1)[0])
            except Exception:
                next_error = float(np.mean(residual_window))
            if week + 1 < horizon:
                forecasts[week + 1] = base_forecast[week + 1] + next_error
        if len(residuals) >= DDPFF_SPC_RUN:
            last_run = np.array(residuals[-DDPFF_SPC_RUN:])
            residual_mean = float(np.mean(residuals))
            if np.all(last_run - residual_mean > 0) or np.all(last_run - residual_mean < 0):
                reset_level = float(np.mean(observed[max(0, week - 4) : week + 1]))
                if week + 1 < horizon:
                    forecasts[week + 1] = reset_level
    return forecasts
