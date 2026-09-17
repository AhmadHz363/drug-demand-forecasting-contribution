"""CAMEO Module A — episodic metric-learning embedding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.cold_start.constants import (
    CAMEO_ANALOG_CANDIDATE_MULT,
    CAMEO_LAUNCH_WEEKS,
    CAMEO_SHAPE_RERANK_WEIGHT,
)


@dataclass(frozen=True)
class CameoInitResult:
    init_level: float
    order: np.ndarray
    weights: np.ndarray
    prior_p_zero: float
    launch_profile: np.ndarray


def analog_launch_expected(series: np.ndarray, n_weeks: int = CAMEO_LAUNCH_WEEKS) -> tuple[float, float]:
    """Expected weekly demand and zero-week fraction from an analog's launch window."""
    window = np.asarray(series[: min(n_weeks, len(series))], dtype=float)
    if len(window) == 0:
        return 0.0, 1.0
    zero_frac = float((window == 0).mean())
    positive = window[window > 0]
    positive_mean = float(positive.mean()) if len(positive) else 0.0
    expected = (1.0 - zero_frac) * positive_mean
    return expected, zero_frac


class MetricNet:
    def __init__(self, d_in: int, d_hidden: int, d_out: int, lr: float = 0.02, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, 0.5, (d_hidden, d_out))
        self.b2 = np.zeros(d_out)
        self.lr = lr

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        out = h @ self.W2 + self.b2
        return out, h, z1

    def embed(self, x: np.ndarray) -> np.ndarray:
        _, h, _ = self.forward(x)
        return h

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        out, h, z1 = self.forward(x)
        err = out - y
        n = x.shape[0]
        g_w2 = h.T @ err / n
        g_b2 = err.mean(axis=0)
        d_h = err @ self.W2.T
        d_z1 = d_h * (1 - np.tanh(z1) ** 2)
        g_w1 = x.T @ d_z1 / n
        g_b1 = d_z1.mean(axis=0)
        self.W1 -= self.lr * g_w1
        self.b1 -= self.lr * g_b1
        self.W2 -= self.lr * g_w2
        self.b2 -= self.lr * g_b2
        return float(np.mean(err**2))

    def to_dict(self) -> dict:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MetricNet":
        net = cls(
            d_in=len(payload["W1"][0]),
            d_hidden=len(payload["b1"]),
            d_out=len(payload["b2"]),
            lr=float(payload.get("lr", 0.02)),
        )
        net.W1 = np.asarray(payload["W1"], dtype=float)
        net.b1 = np.asarray(payload["b1"], dtype=float)
        net.W2 = np.asarray(payload["W2"], dtype=float)
        net.b2 = np.asarray(payload["b2"], dtype=float)
        return net


def demand_shape_features(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    mean = float(s.mean())
    cv = float(s.std() / (mean + 1e-6))
    zero_frac = float((s == 0).mean())
    t = np.arange(len(s))
    trend = float(np.polyfit(t, s, 1)[0]) if len(s) > 3 else 0.0
    if len(s) > 14:
        s0 = s - s.mean()
        ac12 = float((s0[:-12] * s0[12:]).sum() / ((s0**2).sum() + 1e-6))
    else:
        ac12 = 0.0
    return np.array([mean, cv, zero_frac, trend, ac12], dtype=float)


def train_metric_net(
    attrs_scaled: np.ndarray,
    shape_targets: np.ndarray,
    *,
    epochs: int = 600,
    episodic_mask_frac: float = 0.2,
    seed: int = 1,
) -> MetricNet:
    net = MetricNet(attrs_scaled.shape[1], 12, shape_targets.shape[1], lr=0.05, seed=seed)
    n = attrs_scaled.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        mask = rng.random(n) > episodic_mask_frac
        net.train_step(attrs_scaled[mask], shape_targets[mask])
    return net


def cameo_init_forecast(
    new_attr_scaled: np.ndarray,
    hist_embeds: np.ndarray,
    hist_series: list[np.ndarray],
    net: MetricNet,
    topk: int = 5,
) -> CameoInitResult:
    query = net.embed(new_attr_scaled.reshape(1, -1))[0]
    embed_dist = np.linalg.norm(hist_embeds - query, axis=1)

    candidate_k = min(len(hist_series), max(topk, topk * CAMEO_ANALOG_CANDIDATE_MULT))
    candidate_idx = np.argsort(embed_dist)[:candidate_k]

    predicted_shape, _, _ = net.forward(new_attr_scaled.reshape(1, -1))
    predicted_shape = predicted_shape[0]
    hist_shapes = np.array([demand_shape_features(series) for series in hist_series])

    combined: list[tuple[float, int]] = []
    embed_ref = max(float(embed_dist[candidate_idx].max()), 1e-6)
    for idx in candidate_idx:
        shape_dist = float(np.linalg.norm(hist_shapes[idx] - predicted_shape))
        shape_ref = max(float(np.linalg.norm(hist_shapes[candidate_idx] - predicted_shape, axis=1).max()), 1e-6)
        score = (1.0 - CAMEO_SHAPE_RERANK_WEIGHT) * (
            embed_dist[idx] / embed_ref
        ) + CAMEO_SHAPE_RERANK_WEIGHT * (shape_dist / shape_ref)
        combined.append((score, int(idx)))

    combined.sort(key=lambda item: item[0])
    order = np.array([idx for _, idx in combined[:topk]], dtype=int)
    distances = embed_dist[order]
    weights = 1.0 / (distances + 1e-3)
    weights = weights / weights.sum()

    levels: list[float] = []
    zero_fracs: list[float] = []
    for idx in order:
        expected, zero_frac = analog_launch_expected(hist_series[idx])
        levels.append(expected)
        zero_fracs.append(zero_frac)

    init_level = float(np.dot(weights, levels))
    prior_p_zero = float(np.clip(np.dot(weights, zero_fracs), 0.05, 0.95))

    max_len = max(len(hist_series[idx]) for idx in order)
    profile_len = min(max_len, CAMEO_LAUNCH_WEEKS * 3)
    profiles: list[np.ndarray] = []
    for idx in order:
        series = np.asarray(hist_series[idx][:profile_len], dtype=float)
        if len(series) == 0:
            profiles.append(np.zeros(profile_len))
            continue
        if len(series) < profile_len:
            pad_val = float(series[series > 0].mean()) if np.any(series > 0) else 0.0
            series = np.pad(series, (0, profile_len - len(series)), constant_values=pad_val)
        profiles.append(series)
    profile_matrix = np.stack(profiles, axis=0)
    mean_profile = (weights[:, None] * profile_matrix).sum(axis=0)
    upper_profile = np.percentile(profile_matrix, 75, axis=0)
    launch_profile = np.maximum(
        mean_profile * (1.0 - prior_p_zero * 0.5),
        0.55 * mean_profile + 0.45 * upper_profile * (1.0 - prior_p_zero * 0.35),
    )
    if prior_p_zero < 0.65:
        p90 = np.percentile(profile_matrix, 90, axis=0)
        ramp = min(6, len(launch_profile))
        launch_profile[:ramp] = np.maximum(launch_profile[:ramp], p90[:ramp] * 0.85)

    return CameoInitResult(
        init_level=max(init_level, 0.0),
        order=order,
        weights=weights,
        prior_p_zero=prior_p_zero,
        launch_profile=launch_profile,
    )
