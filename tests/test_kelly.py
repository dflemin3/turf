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


if __name__ == "__main__":
    test_kelly()
