#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: David P Fleming

This file contains routines estimating optimal stake sizes, given prior odds and
estimated probabilities of success using the Kelly Criterion. For more information
on the basic principles behind the Kelly Criterion, check out the following:
https://en.wikipedia.org/wiki/Kelly_criterion

"""

# Imports
import numpy as np


__all__ = ["kelly_bet", "american_odds_return", "normalize_money",
           "american_to_decimal", "decimal_to_american"]


################################################################################
#
# Kelly criterion-related functions - maximize the expected log of wealth
#
################################################################################


def kelly_bet(p : float, odds : float, f : float=1, bankroll : float=1.0,
              normalize_return : bool=True) -> float:
    """
    Given odds and the probability of winning the bet,
    compute the optimal fraction of the bankroll to wager in order to maximize
    the expected log of your wealth using the Kelly Criterion

    Parameters
    ----------
    p : float
        Probability of winning the wager
    odds : float
        Odds associated with the given wager
    f : float (optional)
        Fraction of bankroll in Kelly criterion bet to apply. Defaults to 1.
    bankroll : float (optional)
        Size of bankroll used when computing Kelly bet. Defaults to 1 (unit).
    normalize_return : bool (optional)
        Normalize the value to 2 decimals for USD bets? Defauts to True.

    Returns
    -------
    stake : float
        Kelly bet. Note if stake < 0, take the other side, if available.
    """

    # Validate win probability
    err_msg = f"ERROR: p = {p} must be within the range [0,1]."
    assert 0 <= p <= 1, err_msg

    # Convert to decimal odds
    b = american_to_decimal(odds, decimals=None) - 1.0

    # Compute fractional stake via Kelly Criterion
    fstar = p - ((1 - p) / b)

    # Estimate wager given bankroll, fraction
    stake = fstar * f * bankroll

    # Normalize bet value to 2 decimal places for USD wagers?
    if normalize_return:
        stake = normalize_money(stake)

    return stake


################################################################################
#
# Utility functions
#
################################################################################


def normalize_money(val : float, decimals : int=2) -> float:
    """
    Round monetary value to proper base unit, e.g. for USD, round to 2 decimal
    places. If val = 4.335, this function returns 4.34

    Parameters
    ----------
    val : float
        Monetary value, e.g. USD
    decimals : int (optional)
        Number of decimals to consider. Defaults to 2.

    Returns
    -------
    ret : float
        Monetary value rounded to decimals decimal places
    """

    return np.round(val, decimals=decimals)


def american_odds_return(stake : float, odds : float, success : bool,
                         return_net : bool=False, normalize_return : bool=True) -> float:
    """
    Given (American-style) odds, compute the return from the bet for a given stake and
    whether or not the bet was successful.

    Parameters
    ----------
    stake : float
        total amount to stake
    odds : float
        American odds associated with the bet
    success : bool
        whether or not the bet was successful
    return_net : bool (optional)
        Whether or not to return the net return. Defaults to False, e.g. return
        the profit computed as (net_return - stake)
    normalize_return : bool (optional)
        Whether or not to ensure money is normalized to common units, e.g., cannot
        consider amounts of USD below cents (2 decimal places). Defaults to True.

    Returns
    -------
    ret : float
        Either the net return or the profit (net return - stake), depending on
        whether return_net is True or False. Defaults to returning the profit.
    """

    # Validate that stake is positive
    err_msg = f"ERROR: Can only place a positive stake: stake = {stake}"
    assert stake > 0, err_msg

    # Handle + and - odds
    if odds < 0:
        ret = (100.0 / np.fabs(odds)) * stake
    else:
        ret = odds * (stake / 100.0)

    # If the bet was successful...
    if success:
        # Return net return?
        if return_net:
            ret = ret + stake
    # Lost the bet!
    else:
        # Net return?
        if return_net:
            ret = -stake
        # Return profit?
        else:
            ret = 0.0

    # Ensure correct number of decimal places
    if normalize_return:
        ret = normalize_money(ret)
    return ret


def american_to_decimal(odds, decimals=None):
    """
    Convert American odds to decimal odds. By definition, the decimal odds
    returned by this function are used to compute the net return. For example,
    when given odds = +110, this function would return ret = 2.1 as a successful
    wager of $100 would net a profit of $110, that is, a net return of $210.

    Parameters
    ----------
    odds : float
        American odds, e.g. -110 or +110
    decimals : int (optional)
        Decimals for calculated decimal odds. Defaults to None (no rounding).

    Returns
    -------
    ret : float
        Decimal odds
    """

    # Handle + and - odds
    if odds < 0:
        ret = (100.0 / np.fabs(odds)) + 1
    else:
        ret = (odds / 100.0) + 1

    if decimals is not None:
        return np.round(ret, decimals=decimals)
    else:
        return ret


def decimal_to_american(odds, decimals=None):
    """
    Convert decimal odds to American odds. By definition, the decimal odds
    taken by this function are used to compute the net return. For example,
    when given odds = 2.1, this function would return ret = 110.

    Parameters
    ----------
    odds : float
        decimal odds, e.g. 3.1
    american : int (optional)
        Decimal for calculated American odds. Defaults to None, aka no rounding,
        but a useful choice is 0 (so -110.1 would become -110).

    Returns
    -------
    ret : float
        American odds
    """

    # Validate odds
    err_msg = "ERROR: decimalToAmerican requires decimal odds > 1."
    assert odds > 1, err_msg

    # Handle + and - odds
    if odds < 2:
        ret = -100.0 / (odds - 1)
    else:
        ret = 100 * (odds - 1)

    if decimals is not None:
        return np.round(ret, decimals=decimals)
    else:
        return ret
