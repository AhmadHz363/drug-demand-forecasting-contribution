"""CAMEO end-to-end weekly forecast pipeline."""

from __future__ import annotations

import numpy as np

from app.cold_start.cameo.bocpd import BOCPD
from app.cold_start.cameo.metric_net import MetricNet, cameo_init_forecast
from app.cold_start.cameo.zi_tracker import ZIBayesTracker
from app.cold_start.constants import (
    CAMEO_DRIFT_HAZARD,
    CAMEO_DRIFT_THRESHOLD,
    CAMEO_ONLINE_TAU,
)


def _make_tracker(init_level: float, prior_p_zero: float) -> ZIBayesTracker:
    return ZIBayesTracker(
        prior_p_zero=prior_p_zero,
        prior_mu=float(np.log(max(init_level, 1e-3))),
        prior_var=0.5,
        decay=0.75,
    )


def _profile_level(init, t: int) -> float:
    profile = init.launch_profile
    if len(profile) == 0:
        return init.init_level
    idx = min(t, len(profile) - 1)
    return float(profile[idx])


def run_cameo(
    new_attr_scaled: np.ndarray,
    hist_embeds: np.ndarray,
    hist_series: list[np.ndarray],
    net: MetricNet,
    actual_demand: np.ndarray,
    *,
    drift_hazard: float = CAMEO_DRIFT_HAZARD,
    drift_threshold: float = CAMEO_DRIFT_THRESHOLD,
    topk: int = 5,
    tau: float = CAMEO_ONLINE_TAU,
) -> tuple[np.ndarray, list[bool]]:
    init = cameo_init_forecast(new_attr_scaled, hist_embeds, hist_series, net, topk=topk)
    tracker = _make_tracker(init.init_level, init.prior_p_zero)
    bocpd = BOCPD(hazard=drift_hazard)
    forecasts: list[float] = []
    drift_flags: list[bool] = []
    zero_streak = 0

    for t, obs in enumerate(actual_demand):
        mu_online, var_online = tracker.predict()
        profile_level = _profile_level(init, t)
        w_online = (t + 2) / (t + tau + 2)
        yhat = (1.0 - w_online) * profile_level + w_online * mu_online

        if t >= 1 and init.prior_p_zero < 0.7:
            prev = float(actual_demand[t - 1])
            run_mean = float(np.mean(actual_demand[:t]))
            if prev > yhat * 1.15:
                yhat = 0.1 * yhat + 0.9 * prev
            elif run_mean > yhat:
                yhat = 0.25 * yhat + 0.75 * run_mean

        if t >= 1 and init.prior_p_zero < 0.35:
            history = actual_demand[:t]
            hist_mean = float(np.mean(history))
            hist_max = float(np.max(history))
            if hist_mean > yhat:
                yhat = 0.25 * yhat + 0.75 * hist_mean
            if hist_max > yhat * 1.25:
                yhat = 0.4 * yhat + 0.6 * hist_max

        if obs == 0:
            zero_streak += 1
        else:
            zero_streak = 0

        if zero_streak >= 2:
            decay = 0.55 ** (zero_streak - 1)
            yhat = min(yhat, profile_level * decay, mu_online * decay)

        forecasts.append(max(0.0, float(yhat)))
        resid_z = (obs - yhat) / (np.sqrt(var_online) + 1e-6)
        p_cp = bocpd.step(float(resid_z))
        drift_flags.append(p_cp > drift_threshold)
        tracker.update(float(obs))
        if drift_flags[-1]:
            init = cameo_init_forecast(new_attr_scaled, hist_embeds, hist_series, net, topk=topk)
            tracker = _make_tracker(init.init_level, init.prior_p_zero)
            zero_streak = 0

    return np.asarray(forecasts, dtype=float), drift_flags


def conformal_intervals(residuals_calib: np.ndarray, alpha: float = 0.1) -> float:
    return float(np.quantile(np.abs(residuals_calib), 1 - alpha))


def forecast_cold_start_weeks(
    new_attr_scaled: np.ndarray,
    hist_embeds: np.ndarray,
    hist_series: list[np.ndarray],
    net: MetricNet,
    *,
    horizon_weeks: int,
    observed_weeks: np.ndarray | None = None,
    topk: int = 5,
) -> tuple[np.ndarray, float]:
    """
    Forecast ``horizon_weeks`` of weekly demand.

    When ``observed_weeks`` is provided, CAMEO updates online for those weeks
    then extrapolates the last forecast level for any remaining horizon.
    """
    init = cameo_init_forecast(new_attr_scaled, hist_embeds, hist_series, net, topk=topk)

    if observed_weeks is None or len(observed_weeks) == 0:
        profile = init.launch_profile
        if len(profile) >= horizon_weeks:
            return profile[:horizon_weeks].astype(float), init.init_level
        if len(profile) == 0:
            return np.full(horizon_weeks, init.init_level, dtype=float), init.init_level
        tail = np.full(horizon_weeks - len(profile), profile[-1], dtype=float)
        return np.concatenate([profile, tail]), float(profile[-1])

    observed = np.asarray(observed_weeks, dtype=float)
    cameo_forecasts, _ = run_cameo(
        new_attr_scaled,
        hist_embeds,
        hist_series,
        net,
        observed,
        topk=topk,
    )
    if len(cameo_forecasts) >= horizon_weeks:
        return cameo_forecasts[:horizon_weeks], float(cameo_forecasts[-1])

    tail = np.full(horizon_weeks - len(cameo_forecasts), cameo_forecasts[-1], dtype=float)
    return np.concatenate([cameo_forecasts, tail]), float(cameo_forecasts[-1])
