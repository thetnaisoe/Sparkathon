"""Distributions behind the claim ranges.

Shares (0..1) are Beta distributions: fitted to the stated P10/P50/P90 and
updated with Beta-Binomial conjugacy when experiment results arrive.
Magnitudes (counts, prices) use a split lognormal that reproduces the stated
P10, P50 and P90 exactly.

All sampling is by inverse CDF from fixed uniforms (see model.py), so after an
update only the updated driver's draws change and before/after comparisons are
not blurred by fresh sampling noise. (A result below the estimate lowers the
share's mean and median, but a narrower posterior can raise its P10 and, through
the product, occasionally the output's P50; evals/run_evals.py measures this.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize, special, stats

Z90 = 1.2815515655446004  # standard normal quantile at 0.9
PRIOR_MAX_STRENGTH = 20.0  # an unverified assumption counts as at most 20 observations
_EPS = 1e-4
_QS = np.array([0.1, 0.5, 0.9])


@dataclass(frozen=True)
class Beta:
    alpha: float
    beta: float

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def quantiles(self) -> tuple[float, float, float]:
        q = stats.beta.ppf(_QS, self.alpha, self.beta)
        return float(q[0]), float(q[1]), float(q[2])

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return special.betaincinv(self.alpha, self.beta, u)

    def update(self, successes: int, trials: int) -> Beta:
        if not 0 <= successes <= trials:
            raise ValueError("need 0 <= successes <= trials")
        return Beta(self.alpha + successes, self.beta + (trials - successes))


@dataclass(frozen=True)
class SplitLognormal:
    p10: float
    p50: float
    p90: float

    def __post_init__(self) -> None:
        if not 0 < self.p10 <= self.p50 <= self.p90:
            raise ValueError("a magnitude needs 0 < p10 <= p50 <= p90")

    def quantiles(self) -> tuple[float, float, float]:
        return self.p10, self.p50, self.p90

    def ppf(self, u: np.ndarray) -> np.ndarray:
        z = special.ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        mu = math.log(self.p50)
        s_lo = (mu - math.log(self.p10)) / Z90
        s_hi = (math.log(self.p90) - mu) / Z90
        return np.exp(mu + np.where(z < 0, z * s_lo, z * s_hi))


Belief = Beta | SplitLognormal


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, _EPS, 1 - _EPS)
    return np.log(x / (1 - x))


@lru_cache(maxsize=4096)
def fit_beta(p10: float, p50: float, p90: float, *, max_strength: float | None = None) -> Beta:
    """The Beta whose 10/50/90% quantiles best match the stated range.

    Least squares in logit space, started from a moment estimate. With
    `max_strength`, a fit that is more confident than that many observations is
    widened (same mean, smaller alpha + beta), because an unverified assumption
    should not outweigh a real experiment.
    """
    if not 0 <= p10 <= p50 <= p90 <= 1:
        raise ValueError("a share needs 0 <= p10 <= p50 <= p90 <= 1")
    target = np.clip(np.array([p10, p50, p90], dtype=float), _EPS, 1 - _EPS)
    m = float(target[1])
    sd = max(float(target[2] - target[0]) / (2 * Z90), 1e-3)
    var = min(sd**2, m * (1 - m) * 0.9)
    k0 = max(m * (1 - m) / var - 1, 0.5)
    x0 = np.log([max(m * k0, 0.05), max((1 - m) * k0, 0.05)])
    goal = _logit(target)

    def loss(x: np.ndarray) -> float:
        a, b = np.exp(np.clip(x, -6, 12))
        q = stats.beta.ppf(_QS, a, b)
        return float(np.sum((_logit(q) - goal) ** 2))

    res = optimize.minimize(loss, x0, method="Nelder-Mead",
                            options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
    a, b = (float(v) for v in np.exp(np.clip(res.x, -6, 12)))
    if max_strength is not None and a + b > max_strength:
        scale = max_strength / (a + b)
        a, b = a * scale, b * scale
    return Beta(a, b)


def beta_binomial_pmf(trials: int, belief: Beta) -> np.ndarray:
    """Predictive probability of 0..trials successes in the next `trials` trials."""
    ks = np.arange(trials + 1)
    return stats.betabinom.pmf(ks, trials, belief.alpha, belief.beta)
