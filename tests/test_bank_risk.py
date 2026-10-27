import numpy as np
import pytest

from simulations.bank_risk import risk_metrics, simulate_bank_losses

TWO_REGIMES = {
    "good": {"prob": 0.8, "p_default": 0.01, "p_up": 0.51},
    "bad": {"prob": 0.2, "p_default": 0.06, "p_up": 0.46},
}
INDEPENDENT = {"all": {"prob": 1.0, "p_default": 0.02, "p_up": 0.5}}
PORTFOLIO = {"n_loans": 100, "ead": 10_000, "lgd": 0.45, "position_value": 100_000, "n_days": 250}


def test_expected_credit_loss_close_to_9000():
    credit = simulate_bank_losses(TWO_REGIMES, **PORTFOLIO, seed=42)["credit"]
    assert credit.mean() == pytest.approx(9_000, rel=0.02)


def test_regimes_fatten_the_tail():
    var_ind = risk_metrics(
        simulate_bank_losses(INDEPENDENT, **PORTFOLIO, seed=42)["credit"]
    )["VaR"]
    var_reg = risk_metrics(
        simulate_bank_losses(TWO_REGIMES, **PORTFOLIO, seed=42)["credit"]
    )["VaR"]
    assert var_reg > var_ind


def test_reproducible_with_seed():
    a = simulate_bank_losses(TWO_REGIMES, **PORTFOLIO, n_sims=1_000, seed=7)
    b = simulate_bank_losses(TWO_REGIMES, **PORTFOLIO, n_sims=1_000, seed=7)
    np.testing.assert_array_equal(a["total"], b["total"])


def test_regime_probs_must_sum_to_one():
    bad = {"a": {"prob": 0.5, "p_default": 0.01, "p_up": 0.5}}
    with pytest.raises(ValueError):
        simulate_bank_losses(bad, **PORTFOLIO)


def test_risk_metrics_constant_losses():
    metrics = risk_metrics(np.full(1_000, 500.0))
    assert metrics == pytest.approx({"EL": 500.0, "VaR": 500.0, "ES": 500.0})
