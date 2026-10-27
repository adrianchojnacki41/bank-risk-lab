import pytest

from simulations.credit import (
    default_count_pmf,
    expected_credit_loss,
    prob_at_least_k_defaults,
    prob_at_least_one_default,
)


def test_two_loans_hand_calculation():
    assert default_count_pmf(2, 0.2) == pytest.approx([0.64, 0.32, 0.04])


def test_pmf_sums_to_one():
    assert default_count_pmf(100, 0.02).sum() == pytest.approx(1.0)


def test_at_least_k_one_matches_at_least_one():
    assert prob_at_least_k_defaults(100, 0.02, 1) == pytest.approx(
        prob_at_least_one_default(100, 0.02)
    )


def test_expected_credit_loss():
    assert expected_credit_loss(100, 0.02, 10_000, 0.45) == pytest.approx(9_000)


def test_negative_loans_raise():
    with pytest.raises(ValueError):
        default_count_pmf(-1, 0.02)
