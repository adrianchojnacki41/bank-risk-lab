import numpy as np

from simulations.credit import _default_for_x_clients
from simulations.validation import check_count, check_non_negative, check_probability


def stock_price_distribution(
    s0: float, n_days: int, p_up: float, up: float = 0.01, down: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Possible stock prices after n_days in a binomial model and their probabilities."""
    check_non_negative(s0, "s0")
    check_count(n_days, "n_days")
    check_probability(p_up, "p_up")
    check_probability(up, "up")
    check_probability(down, "down")

    prices = []
    probs = []
    for k in range(n_days + 1):
        prices.append(s0 * (1 + up) ** k * (1 - down) ** (n_days - k))
        probs.append(_default_for_x_clients(n_days, p_up, k))

    return np.array(prices), np.array(probs)
