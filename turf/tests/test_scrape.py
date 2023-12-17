#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test turf data scraping functions

@author: David Fleming, 2023

"""

from turf import scrape
import numpy as np


def test_full_nfl_season_raw_scrape():
    """
    Test data full season raw data scraping function

    Parameters
    ----------
    None

    Returns
    -------
    """

    # Check to see if you pull full season raw, data is correct
    truth = ['2020-02-02', 'SuperBowl', 'KC', '31', 'SF', '20']
    test = scrape.pull_nfl_full_season_games_raw(year=2019).iloc[-1].values.tolist()
    err_msg = "Error in scrape.pull_full_season_games_raw test"
    assert np.all(truth == test), err_msg


def test_NFLSeason_data():
    """
    Test to ensure NFLSeason class scrapes and processes data as expected.
    """



    # Pull data
    season = scrape.NFLSeason(year=2020, week=None)

    # Test 0) year and week
    err_msg = "Error in test 0 for test_Season_data - incorrect season/year"
    assert season.year == 2020, err_msg
    err_msg = "Error in test 0 for test_Season_data - incorrect season/year"
    assert season.week is None, err_msg

    # Test 1) Pull all season games, process data
    # Make sure unplayed dataframe is empty (full season has been played)
    err_msg = "Error in test 1 for test_Season_data - incorrect unplayed_df"
    assert season.unplayed_df.empty, err_msg

    # Test 2) Check full schedule
    test = season.full_schedule.iloc[15].values.tolist()
    err_msg = "Error in test 2 for test_Season_data - incorrect full schedule"
    assert np.all(test == ['2020-09-14', '1', 'DEN', 'TEN']), err_msg

    # Test 3) Check played df
    test = season.played_df.iloc[25].values.tolist()
    err_msg = "Error in test 3 for test_Season_week_data - incorrect played_df"
    assert(np.all(test == ['2020-09-20', '2', 'NYJ', '13', 'SF', '31'])), err_msg


def test_NFLSeason_week_data():
    """
    Test to ensure Season class scrapes and processes data as expected.
    """

    # Pull data as if season has only been played through week 6
    season = scrape.NFLSeason(year=2018, week=7)

    # Test 0) year and week
    err_msg = "Error in test 0 for test_Season_week_data - incorrect season/year"
    assert season.year == 2018, err_msg
    err_msg = "Error in test 0 for test_Season_week_data - incorrect season/year"
    assert season.week == 7, err_msg

    # Test 1) Pull all season games, process data
    # Make sure unplayed dataframe is empty (full season has been played)
    err_msg = "Error in test 1 for test_Season_week_data - incorrect unplayed_df"
    test = season.unplayed_df.iloc[7].values.tolist()
    assert np.all(test == ['2018-10-21', '7', 'JAX', np.nan, 'HST', np.nan]), err_msg

    # Test 2) Check full schedule
    test = season.full_schedule.iloc[111].values.tolist()
    err_msg = "Error in test 2 for test_Season_week_data - incorrect full schedule"
    assert np.all(test == ['2018-10-28', '8', 'DET', 'SEA']), err_msg

    # Test 3) Check played df
    test = season.played_df.iloc[90].values.tolist()
    err_msg = "Error in test 3 for test_Season_week_data - incorrect played_df"
    assert(np.all(test == ['2018-10-14', '6', 'TEN', '0', 'BLT', '21'])), err_msg


if __name__ == "__main__":
    test_full_nfl_season_raw_scrape()
    test_NFLSeason_data()
    test_NFLSeason_week_data()
