#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test turf name parsing

@author: David Fleming, 2023

"""

from turf.utils import _nfl_name_conv, _nhl_name_conv


def test_nfl_name_parser():
    """
    Test football name parsing dictionary

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1 name parsing
    name = "Washington Football Team"
    parsed_name = _nfl_name_conv[name]

    err_msg = "Test 1 failure: name _nfl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "WAS", err_msg

    # Test 2 name parsing
    name = "Washington Redskins"
    parsed_name = _nfl_name_conv[name]

    err_msg = "Test 2 failure: name _nfl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "WAS", err_msg

    # Test 3 name parsing
    name = "Arizona Cardinals"
    parsed_name = _nfl_name_conv[name]

    err_msg = "Test 3 failure: name _nfl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "ARZ", err_msg

    # Test 4 name parsing
    name = "St. Louis Rams"
    parsed_name = _nfl_name_conv[name]

    err_msg = "Test 4 failure: name _nfl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "LAR", err_msg


def test_nhl_name_parser():
    """
    Test hockey name parsing dictionary

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Test 1 name parsing
    name = "St. Louis Blues"
    parsed_name = _nhl_name_conv[name]

    err_msg = "Test 1 failure: name _nhl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "STL", err_msg

    # Test 2 name parsing
    name = "Saint Louis Blues"
    parsed_name = _nhl_name_conv[name]

    err_msg = "Test 2 failure: name _nhl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "STL", err_msg

    # Test 3 name parsing
    name = "Vegas Golden Knights"
    parsed_name = _nhl_name_conv[name]

    err_msg = "Test 3 failure: name _nhl_name_conv returned incorrect abbreviated team name"
    assert parsed_name == "VGK", err_msg


if __name__ == "__main__":
    test_nfl_name_parser()
    test_nhl_name_parser()
