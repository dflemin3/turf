#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test Kelly bets

@author: David P. Fleming, 2024
"""

from turf import bet
import numpy as np


def test_kelly():
    """
    Test Kelly criterion functions

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1 - even money stakes
    p = 0.6
    odds = 100
    f = 1
    bankroll = 1
    normalize_return = True
    truth = 0.2
    test = bet.kelly_bet(p, odds, f, bankroll,
                         normalize_return=normalize_return)

    err_msg = f"Test 1 kelly_bet failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 2 - underdog
    p = 0.54
    odds = 200
    f = 1
    bankroll = 1
    normalize_return = True
    truth = 0.31
    test = bet.kelly_bet(p, odds, f, bankroll,
                         normalize_return=normalize_return)

    err_msg = f"Test 2 kelly_bet failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 3 - Favorite
    p = 0.95
    odds = -250
    f = 1
    bankroll = 1
    normalize_return = True
    truth = 0.82
    test = bet.kelly_bet(p, odds, f, bankroll,
                         normalize_return=normalize_return)

    err_msg = f"Test 3 kelly_bet failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 4 - Favorite, hedging bet
    p = 0.95
    odds = -250
    f = 0.3
    bankroll = 1
    normalize_return = True
    truth = 0.25
    test = bet.kelly_bet(p, odds, f, bankroll,
                         normalize_return=normalize_return)

    err_msg = f"Test 4 kelly_bet failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 5 - Underdog, hedging bet
    p = 0.63
    odds = 133
    f = 0.4
    bankroll = 1
    normalize_return = True
    truth = 0.14
    test = bet.kelly_bet(p, odds, f, bankroll,
                         normalize_return=normalize_return)

    err_msg = f"Test 5 kelly_bet failure: - truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg


if __name__ == "__main__":
    test_kelly()
