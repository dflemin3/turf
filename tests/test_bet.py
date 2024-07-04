#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test estimating returns and profits from wagers 

@author: David P. Fleming, 2024
"""

from turf import bet
import numpy as np


def test_returns():
    """
    Test American odds wager return functions

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1: - odds, successful bet, profit
    odds = -129
    stake = 13.75
    success = 1
    truth = 10.66
    test = bet.american_odds_return(stake, odds, success, return_net=False)

    err_msg = f"Test 1 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 2: - odds, successful bet, net return
    odds = -129
    stake = 13.75
    success = 1
    truth = 24.41
    test = bet.american_odds_return(stake, odds, success, return_net=True)

    err_msg = f"Test 2 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 3: - odds, unsuccessful bet, profit
    odds = -129
    stake = 13.75
    success = 0
    truth = 0
    test = bet.american_odds_return(stake, odds, success, return_net=False)

    err_msg = f"Test 3 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 4: - odds, unsuccessful bet, net return
    odds = -129
    stake = 13.75
    success = 0
    truth = -stake
    test = bet.american_odds_return(stake, odds, success, return_net=True)

    err_msg = f"Test 4 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 5: + odds, successful bet, profit
    odds = 119
    stake = 15.0
    success = 1
    truth = 17.85
    test = bet.american_odds_return(stake, odds, success, return_net=False)

    err_msg = f"Test 5 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 6: + odds, successful bet, net return
    odds = 119
    stake = 15.0
    success = 1
    truth = 32.85
    test = bet.american_odds_return(stake, odds, success, return_net=True)

    err_msg = f"Test 6 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 7: + odds, unsuccessful bet, profit
    odds = 119
    stake = 15.0
    success = 0
    truth = 0
    test = bet.american_odds_return(stake, odds, success, return_net=False)

    err_msg = f"Test 7 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 8: - odds, unsuccessful bet, net return
    odds = 119
    stake = 15.0
    success = 0
    truth = -stake
    test = bet.american_odds_return(stake, odds, success, return_net=True)

    err_msg = f"Test 8 american_odds_return failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg
# end function


def test_normalize_money():
    """
    Test function that normalizes monetary value to proper number of decimals,
    e.g. smallest USD unit is 1 cent, so only 2 decimal places are permitted.
    """

    # Test
    val = -11.78583323242
    true = -11.79
    test = bet.normalize_money(val)

    err_msg = f"Test 1 normalize_money failure: - true = {true}, test = {test}"
    assert np.allclose(true, test), err_msg

    val = -11.7842555555
    true = -11.78
    test = bet.normalize_money(val)

    err_msg = f"Test 2 normalize_money failure: - true = {true}, test = {test}"
    assert np.allclose(true, test), err_msg

    val = 2.3467e1
    true = 2.347e1
    test = bet.normalize_money(val)

    err_msg = f"Test 3 normalize_money failure: - true = {true}, test = {test}"
    assert np.allclose(true, test), err_msg

    val = 2.33333333
    true = 2.33
    test = bet.normalize_money(val)

    err_msg = f"Test 4 normalize_money failure: - true = {true}, test = {test}"
    assert np.allclose(true, test), err_msg

# end function

if __name__ == "__main__":
    test_returns()
    test_normalize_money()
