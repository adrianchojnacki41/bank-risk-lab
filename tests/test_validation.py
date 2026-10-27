import numpy as np
import pytest

from simulations.validation import check_count, check_non_negative, check_probability


def test_valid_values_pass():
    check_count(5, "n")
    check_probability(0.02, "p")
    check_non_negative(10_000, "ead")


@pytest.mark.parametrize("value", [2.5, True, "100"])
def test_count_rejects_wrong_type(value):
    with pytest.raises(TypeError):
        check_count(value, "n")


@pytest.mark.parametrize("value", [1.5, -0.1, np.nan])
def test_probability_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        check_probability(value, "p")


def test_non_negative_rejects_negative():
    with pytest.raises(ValueError):
        check_non_negative(-100, "ead")
