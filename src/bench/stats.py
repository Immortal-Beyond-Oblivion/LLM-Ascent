"""Statistics utilities: mean, sample std, Wilson score intervals, bootstrap.

The Wilson interval is used for every accuracy claim in the report.
It is a confidence interval for a binomial proportion that performs well
even with small n or extreme p (unlike the normal approximation).

``bootstrap_mean_ci`` covers paired contrasts, where the observations are
differences per item and the resampling unit must be the item.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(values) < 2:
        raise ValueError("std requires at least two values")
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value for Student's t with `df` degrees of freedom.

    Computed from the regularised incomplete beta function via a continued
    fraction, so no SciPy dependency. p = I_{df/(df+t^2)}(df/2, 1/2).
    """
    x = df / (df + t * t)
    return _regularised_beta(x, df / 2.0, 0.5)


def _regularised_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b + lbeta) / a

    # Lentz's algorithm for the continued fraction.
    tiny = 1e-30
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        step = c * d
        f *= step
        if abs(step - 1.0) < 3e-16:
            break

    if x < (a + 1.0) / (a + b + 2.0):
        return front * (f - 1.0)
    # Symmetry identity: I_x(a,b) = 1 - I_{1-x}(b,a).
    return 1.0 - _regularised_beta(1.0 - x, b, a)


def paired_ttest(
    treatment: Sequence[float], control: Sequence[float]
) -> dict[str, float]:
    """Two-sided paired t-test on matched samples.

    Use this ONLY for genuinely paired runs — the same seed, data order and
    corpus for both arms. An unpaired test on two independently seeded sweeps
    throws away the pairing and inflates the variance.

    Always reports the TWO-SIDED p-value. A one-sided p may only be reported if
    the direction was fixed BEFORE the data was seen; retrofitting a
    directional hypothesis after an observed non-significant two-sided result is
    p-hacking, and this project does not do it.

    Returns mean_diff (treatment - control), sd of the differences, t, df, and
    the two-sided p-value.
    """
    if len(treatment) != len(control):
        raise ValueError("paired t-test requires equal-length samples")
    n = len(treatment)
    if n < 2:
        raise ValueError("paired t-test requires at least two pairs")
    diffs = [a - b for a, b in zip(treatment, control)]
    md = mean(diffs)
    sd = std(diffs)
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else (0.0 if md == 0 else math.inf)
    df = n - 1
    return {
        "mean_diff": md,
        "sd": sd,
        "se": se,
        "t": t,
        "df": float(df),
        "n": float(n),
        "p_value": _t_two_sided_p(t, df),
    }


def mcnemar_exact(b: int, c: int) -> dict:
    """Exact McNemar test for paired binary outcomes.

    ``b`` and ``c`` are the DISCORDANT pair counts, not totals:

        b = items where condition A succeeded and condition B failed
        c = items where condition B succeeded and condition A failed

    Pairs where both conditions agree carry no information about the
    difference and are conditioned out, which is what makes this the right
    test for "did the answer change between conditions" on the same items.

    Two-sided p = 2 * P(X <= min(b, c)) with X ~ Binomial(b + c, 1/2).
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must not be negative")
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_value": 1.0,
                "note": "no discordant pairs: the conditions never disagreed"}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": b, "c": c, "discordant": n,
            "p_value": min(1.0, 2.0 * tail)}


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the mean of paired differences.

    The resampling unit is the ITEM. Repeated generations of the same item
    share that item's difficulty and are not independent draws, so resampling
    them as if they were would understate the interval. Resampling items keeps
    the within-item correlation intact.

    Deterministic given ``seed`` (stdlib ``random``), so a reported interval
    can be reproduced exactly.
    """
    if not values:
        raise ValueError("bootstrap_mean_ci requires at least one value")
    n = len(values)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_i = max(0, int(math.floor(alpha * n_resamples)))
    hi_i = min(n_resamples - 1, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)
    return {
        "mean": mean(values),
        "ci_low": means[lo_i],
        "ci_high": means[hi_i],
        "n": n,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "method": "percentile bootstrap, resampling items",
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes (must be 0 <= successes <= total).
        total: Total number of trials (must be > 0).
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) as floats in [0, 1].

    The Wilson interval is:
        center = (p + z²/(2n)) / (1 + z²/n)
        half_width = (z / (1 + z²/n)) * sqrt(p(1-p)/n + z²/(4n²))
    where p = successes/n and z is the two-sided normal quantile.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be in [0, total]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    p = successes / total
    z = _z_score(confidence)
    z2 = z * z
    n = total

    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return (lower, upper)


def _z_score(confidence: float) -> float:
    """Two-sided normal quantile for the given confidence level.

    Uses the inverse error function: z = sqrt(2) * erf_inv(2*confidence - 1).
    """
    # For common confidence levels, use exact values to avoid approximation error
    known = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    if confidence in known:
        return known[confidence]

    # General case: z = sqrt(2) * erf_inv(2*confidence - 1)
    # Using the relationship between erf and the normal CDF
    p = 2 * confidence - 1
    # Abramowitz and Stegun approximation for erf_inv
    # (accurate to ~1e-4, sufficient for non-standard confidence levels)
    a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
    y = math.log(1 - p * p)
    z = math.sqrt(2) * math.copysign(
        math.sqrt(
            math.sqrt((2 / (math.pi * a) + y / 2) ** 2 - y / a)
            - (2 / (math.pi * a) + y / 2)
        ),
        p,
    )
    return z
