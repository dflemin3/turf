"""

Test turf data scraping functions

@author: David Fleming, 2024

"""

import numpy as np
import pandas as pd
import os
from importlib import resources
from turf import scrape


def test_NFLSeason_data():
    """
    Test to ensure NFLSeason class scrapes and processes data as expected.
    """

    # Pull data locally
    path = os.path.join(resources.files("turf.data"), "nfl_sample.csv")
    season = scrape.NFLSeason(year=2020, path=path)

    # Test 0) year
    err_msg = "Error in test 0 for test_Season_data - incorrect season/year"
    assert season.year == 2020, err_msg

    # Test 1) Pull all season games, process data
    # Make sure unplayed dataframe is empty (full season has been played)
    err_msg = "Error in test 1 for test_NFLSeason_data - incorrect unplayed_df"
    assert season.unplayed_df.empty, err_msg

    # Test 2) Check full schedule
    test = season.full_schedule.iloc[15].values
    err_msg = "Error in test 2 for test_NFLSeason_data - incorrect full schedule"
    assert np.all(np.asarray([pd.to_datetime('2020-09-14', format='%Y-%m-%d'),
                              '1', 
                              'DEN', 
                              'TEN']) == test), err_msg

    # Test 3) Check played df
    test = season.played_df.iloc[25].values.tolist()
    err_msg = "Error in test 3 for test_NFLSeason_data - incorrect played_df"
    assert np.all(np.asarray([pd.to_datetime('2020-09-20', format='%Y-%m-%d'),
                              '2', 
                              'NYJ', 
                              13.0, 
                              'SF', 
                              31.0,
                              False]) == test), err_msg


def test_NHLSeason_data():
    """
    Test to ensure NHLSeason class scrapes and processes data as expected.
    """

    # Pull data locally
    path = os.path.join(resources.files("turf.data"), "nhl_sample.csv")
    season = scrape.NHLSeason(year=2018, path=path)

    # Test 0) year
    err_msg = "Error in test 0 for test_NHLSeason_data - incorrect season/year"
    assert season.year == 2018, err_msg

    # Test 1) Pull all season games, process data
    # Make sure unplayed dataframe is empty (full season has been played)
    err_msg = "Error in test 1 for test_NHLSeason_data - incorrect unplayed_df"
    assert season.unplayed_df.empty, err_msg

    # Test 2) Check played df
    test = season.played_df.iloc[653].values.tolist()
    err_msg = "Error in test 2 for test_NHLSeason_data - incorrect played_df"
    assert np.all(np.asarray([pd.to_datetime('2019-01-07', format='%Y-%m-%d'),
                              'PHI', 
                              0, 
                              'STL', 
                              3, 
                              False]) == test), err_msg


if __name__ == "__main__":
    test_NFLSeason_data()
    test_NHLSeason_data()