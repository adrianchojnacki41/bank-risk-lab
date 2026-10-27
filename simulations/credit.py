import math

import numpy as np

from simulations.validation import check_count, check_non_negative, check_probability


def _default_for_x_clients(n_loans: int, p_default: float, x: int) -> float:
    """Probability that exactly x out of n_loans clients default (no input checks)."""
    n_ways = math.comb(n_loans, x)
    return n_ways * p_default**x * (1 - p_default) ** (n_loans - x)


def prob_at_least_one_default(n_loans: int, p_default: float) -> float:
    """Probability of at least one default among n_loans independent loans."""
    check_count(n_loans, "n_loans")
    check_probability(p_default, "p_default")
    prob_no_default = _default_for_x_clients(n_loans, p_default, 0)
    return 1 - prob_no_default


def default_count_pmf(n_loans: int, p_default: float) -> np.ndarray:
    """Probability of exactly k defaults for each k = 0, ..., n_loans."""
    check_count(n_loans, "n_loans")
    check_probability(p_default, "p_default")
    probs = []
    for k in range(n_loans + 1):
        probs.append(_default_for_x_clients(n_loans, p_default, k))
    return np.array(probs)


def prob_at_least_k_defaults(n_loans: int, p_default: float, k: int) -> float:
    """Probability of at least k defaults among n_loans independent loans."""
    check_count(k, "k")
    pmf = default_count_pmf(n_loans, p_default)
    return float(pmf[k:].sum())


def expected_credit_loss(n_loans: int, p_default: float, ead: float, lgd: float) -> float:
    """Expected credit loss of n_loans identical loans: n * PD * EAD * LGD."""
    check_count(n_loans, "n_loans")
    check_probability(p_default, "p_default")
    check_non_negative(ead, "ead")
    check_probability(lgd, "lgd")
    return n_loans * p_default * ead * lgd
