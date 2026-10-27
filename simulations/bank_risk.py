import numpy as np

from simulations.validation import check_count, check_non_negative, check_probability


def _check_regimes(regimes: dict) -> None:
    """Check that regimes have the required keys, valid probabilities and sum to 1."""
    if not isinstance(regimes, dict) or len(regimes) == 0:
        raise TypeError("regimes must be a non-empty dict")
    for name, params in regimes.items():
        for key in ("prob", "p_default", "p_up"):
            if key not in params:
                raise ValueError(f"regime '{name}' is missing key '{key}'")
            check_probability(params[key], f"{name}.{key}")
    total = sum(params["prob"] for params in regimes.values())
    if not np.isclose(total, 1):
        raise ValueError(f"regime probabilities must sum to 1, got {total}")


def simulate_bank_losses(
    regimes: dict,
    n_loans: int,
    ead: float,
    lgd: float,
    position_value: float,
    n_days: int,
    up: float = 0.01,
    down: float = 0.01,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate annual bank losses (credit, market, total) under economic regimes."""
    _check_regimes(regimes)
    check_count(n_loans, "n_loans")
    check_non_negative(ead, "ead")
    check_probability(lgd, "lgd")
    check_non_negative(position_value, "position_value")
    check_count(n_days, "n_days")
    check_probability(up, "up")
    check_probability(down, "down")
    check_count(n_sims, "n_sims")

    rng = np.random.default_rng(seed)

    names = list(regimes)
    probs = [regimes[name]["prob"] for name in names]
    regime_idx = rng.choice(len(names), size=n_sims, p=probs)
    pd_scenario = np.array([regimes[name]["p_default"] for name in names])[regime_idx]
    p_up_scenario = np.array([regimes[name]["p_up"] for name in names])[regime_idx]

    n_defaults = rng.binomial(n_loans, pd_scenario)
    credit = n_defaults * ead * lgd

    n_ups = rng.binomial(n_days, p_up_scenario)
    price_ratio = (1 + up) ** n_ups * (1 - down) ** (n_days - n_ups)
    market = position_value * (1 - price_ratio)

    return {"credit": credit, "market": market, "total": credit + market}


def risk_metrics(losses: np.ndarray, alpha: float = 0.99) -> dict[str, float]:
    """Expected loss, Value at Risk and Expected Shortfall of a loss distribution."""
    check_probability(alpha, "alpha")
    losses = np.asarray(losses)
    var = np.quantile(losses, alpha)
    return {
        "EL": float(losses.mean()),
        "VaR": float(var),
        "ES": float(losses[losses >= var].mean()),
    }
