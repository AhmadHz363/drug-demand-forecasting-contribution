"""CAMEO Module C — Bayesian online changepoint detection."""

from __future__ import annotations

import numpy as np


def _t_pdf(x: float, mu: float, var: float, kappa: float) -> float:
    v = var / kappa + 1e-6
    return float(np.exp(-0.5 * (x - mu) ** 2 / v) / np.sqrt(2 * np.pi * v))


class BOCPD:
    def __init__(self, hazard: float = 1 / 25, max_run: int = 60):
        self.hazard = hazard
        self.max_run = max_run
        self.run_probs = np.array([1.0])
        self.mus = [0.0]
        self.vars_ = [1.0]
        self.kappas = [1.0]

    def step(self, x: float) -> float:
        hazard = self.hazard
        pred = np.array(
            [_t_pdf(x, self.mus[i], self.vars_[i], self.kappas[i]) for i in range(len(self.run_probs))]
        )
        growth = self.run_probs * pred * (1 - hazard)
        cp = float(np.sum(self.run_probs * pred * hazard))
        new_rp = np.concatenate(([cp], growth))
        new_rp /= new_rp.sum() + 1e-12
        new_mu = [0.0] + [
            (self.kappas[i] * self.mus[i] + x) / (self.kappas[i] + 1)
            for i in range(len(self.mus))
        ]
        new_k = [1.0] + [k + 1 for k in self.kappas]
        new_var = [1.0] + [
            self.vars_[i] + (x - self.mus[i]) ** 2 * self.kappas[i] / (self.kappas[i] + 1)
            for i in range(len(self.vars_))
        ]
        if len(new_rp) > self.max_run:
            new_rp = new_rp[: self.max_run] / new_rp[: self.max_run].sum()
            new_mu = new_mu[: self.max_run]
            new_k = new_k[: self.max_run]
            new_var = new_var[: self.max_run]
        self.run_probs, self.mus, self.kappas, self.vars_ = new_rp, new_mu, new_k, new_var
        return float(new_rp[0])
