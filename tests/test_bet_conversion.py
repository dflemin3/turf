#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test functions that convert between American and decimal odds, and vice versa.

@author: David P. Fleming, 2024

"""

from turf import bet
import numpy as np


def test_American_to_decimal():
    """
    Test function to convert American to decimal odds

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1: - odds
    odds = -130
    truth = 1.769
    test = bet.american_to_decimal(odds, decimals=3)

    err_msg = f"Test 1 failure: american_to_decimal returned incorrect odds. Truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 2: + odds
    odds = 105
    truth = 2.05
    test = bet.american_to_decimal(odds, decimals=3)

    err_msg = f"Test 2 failure: american_to_decimal returned incorrect odds. Truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg
# end function


def test_decimal_to_american():
    """
    Test function to convert decimal to American odds

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1: - odds
    odds = 1.909
    truth = -110
    test = bet.decimal_to_american(odds, decimals=0)

    err_msg = f"Test 1 failure: decimal_to_american returned incorrect odds. Truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg

    # Test 2: + odds
    odds = 2.2
    truth = 120
    test = bet.decimal_to_american(odds, decimals=0)

    err_msg = f"Test 2 failure: decimal_to_american returned incorrect odds. Truth = {truth}, test = {test}"
    assert np.allclose(truth, test, equal_nan=False), err_msg
# end function


if __name__ == "__main__":
    test_american_to_decimal()
    test_decimal_to_american()
