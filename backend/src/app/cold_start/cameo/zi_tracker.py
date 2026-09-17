"""CAMEO Module B — zero-inflated Bayesian sequential residual tracker."""

from __future__ import annotations

import numpy as np


class ZIBayesTracker:
    def __init__(
        self,
        prior_p_zero: float = 0.3,
        prior_mu: float = 1.0,
        prior_var: float = 0.5,
        decay: float = 0.85,
    ):
        self.a = (1 - prior_p_zero) * 10 + 1
        self.b = prior_p_zero * 10 + 1
        self.mu = prior_mu
        self.var = prior_var
        self.decay = decay

    def predict(self) -> tuple[float, float]:
        p_zero = self.b / (self.a + self.b)
        expected = (1 - p_zero) * np.exp(self.mu + self.var / 2)
        var_total = expected**2 * (np.exp(self.var) - 1) + p_zero * (1 - p_zero) * np.exp(
            2 * self.mu + self.var
        )
        return float(expected), float(max(var_total, 1e-6))

    def update(self, obs: float) -> None:
        decay = self.decay
        self.a = decay * self.a + (1.0 if obs > 0 else 0.0)
        self.b = decay * self.b + (1.0 if obs == 0 else 0.0)
        if obs > 0:
            x = np.log(obs + 1e-6)
            lr = 1 - decay
            self.mu = (1 - lr) * self.mu + lr * x
            self.var = max((1 - lr) * self.var + lr * (x - self.mu) ** 2, 0.05)
