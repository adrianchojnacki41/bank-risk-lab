import pytest

from simulations.market import stock_price_distribution


def test_two_days():
    prices, probs = stock_price_distribution(100, 2, 0.5)
    assert prices == pytest.approx([98.01, 99.99, 102.01])
    assert probs == pytest.approx([0.25, 0.5, 0.25])


def test_probs_sum_to_one():
    _, probs = stock_price_distribution(100, 250, 0.46)
    assert probs.sum() == pytest.approx(1.0)


def test_bad_p_up_raises():
    with pytest.raises(ValueError):
        stock_price_distribution(100, 10, 1.5)
