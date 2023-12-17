#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test turf data scraping functions

@author: David Fleming, 2023

"""

import numpy as np
import os
from turf import scrape


def test_NFLSeason_data():
    """
    Test to ensure NFLSeason class scrapes and processes data as expected.
    """

    # Pull data locally
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, 'test_data', 'nfl_sample.csv')
    season = scrape.NFLSeason(year=2020, path=path)

    # Test 0) year and week
    err_msg = "Error in test 0 for test_Season_data - incorrect season/year"
    assert season.year == 2020, err_msg

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
    assert(np.all(test == ['2020-09-20', '2', 'NYJ', 13, 'SF', 31])), err_msg


if __name__ == "__main__":
    test_NFLSeason_data()