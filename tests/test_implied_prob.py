"""

Test calculating the implied probability of an event
given odds

@author: David P. Fleming, 2024
"""

from turf import bet
import numpy as np


def test_american_implied_prob():
    """
    Test implied probability calculations
    for american odds

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1: - odds
    odds = -500
    truth = 0.833333333333333
    test = bet.american_implied_probability(odds)

    err_msg = f"Test 1 - odds = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 2: + odds
    odds = -450
    truth = 0.818181818181818
    test = bet.american_implied_probability(odds)

    err_msg = f"Test 2 + odd: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg


if __name__ == "__main__":
    test_american_implied_prob()
