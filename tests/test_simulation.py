"""

Test simulation methods in turf

@author: David Fleming, 2024

"""


import numpy as np
import os
from importlib import resources
from turf import scrape, inference


def test_nb():
    """
    Test game simulation with the NegativeBinomial model
    using 2020 NFL data

    NB:

    ---PHI at SEA---
    O/U: 45 - Over odds : 64.89%
    Spread: SEA 3 - odds of SEA cover : 75.98%
    ML: - odds of SEA ML : 68.85%
    Median outcome: PHI 21 | SEA 29

    Poisson:

    O/U: 45 - Over odds : 80.34%
    Spread: SEA 3 - odds of SEA cover : 93.67%
    ML: - odds of SEA ML : 87.14%
    Median outcome: PHI 21 | SEA 30

    Parameters
    ----------
    None

    Returns
    -------
    """

    # Pull season results to-date
    path = os.path.join(resources.files("turf.data"), "nfl_sample.csv")
    season = scrape.NFLSeason(year=2020, path=path)

    # Initialize model
    path = os.path.join(resources.files("turf.data"), "nfl_example_nb_trace.nc")
    model = inference.IndependentNegativeBinomialMixture(season, path=path)

    # Simulate game
    away_teams = ["PHI"]
    home_teams = ["SEA"]
    ous = [45]
    home_spread = [3]
    for home_team, away_team, ou, hs, ii, in zip(home_teams, away_teams, ous, home_spread, np.arange(len(home_teams))):

        # Simulate n game outcomes
        home_pts, away_pts, home_win, tie = model.simulate_game(home_team, away_team, n=100, seed=1)
        
        total = home_pts + away_pts
        odds = np.mean(total > ou)
        cover = np.mean(home_pts - away_pts > -hs)
        home_ml = np.mean(home_pts > away_pts)

    # Tests to ensure results are similar enough
    # Wide errorbars given short chains, so just check that
    # minimum is exceeded
    
    # Test 0) OU
    err_msg = "Error in test 0 for test_nb - incorrect over odds"
    assert 55 <= np.round(100*odds, decimals=2), err_msg

    # Test 1) ML
    err_msg = "Error in test 1 for test_nb - incorrect over ML"
    assert 60 <= np.round(100*home_ml, decimals=2), err_msg

    # Test 2) Cover
    err_msg = "Error in test 1 for test_nb - incorrect over Cover"
    assert 65 <= np.round(100*cover, decimals=2), err_msg

    
def test_poisson():
    """
    Test game simulation with the Poisson model
    using 2020 NFL data

    Poisson:

    O/U: 45 - Over odds : 80.34%
    Spread: SEA 3 - odds of SEA cover : 93.67%
    ML: - odds of SEA ML : 87.14%
    Median outcome: PHI 21 | SEA 30

    Parameters
    ----------
    None

    Returns
    -------
    """

    # Pull season results to-date
    path = os.path.join(resources.files("turf.data"), "nfl_sample.csv")
    season = scrape.NFLSeason(year=2020, path=path)

    # Initialize model
    path = os.path.join(resources.files("turf.data"), "nfl_example_poisson_trace.nc")
    model = inference.IndependentPoisson(season, path=path)

    # Simulate game
    away_teams = ["PHI"]
    home_teams = ["SEA"]
    ous = [45]
    home_spread = [3]
    for home_team, away_team, ou, hs, ii, in zip(home_teams, away_teams, ous, home_spread, np.arange(len(home_teams))):

        # Simulate n game outcomes
        home_pts, away_pts, home_win, tie = model.simulate_game(home_team, away_team, n=100, seed=None)
        
        total = home_pts + away_pts
        odds = np.mean(total > ou)
        cover = np.mean(home_pts - away_pts > -hs)
        home_ml = np.mean(home_pts > away_pts)

    # Tests to ensure results are similar enough
    # Wide errorbars given short chains, so just check that
    # minimum is exceeded
    
    # Test 0) OU
    err_msg = "Error in test 0 for test_poisson - incorrect over odds"
    assert 65 <= np.round(100*odds, decimals=2), err_msg

    # Test 1) ML
    err_msg = "Error in test 1 for test_poisson - incorrect over ML"
    assert 70 <= np.round(100*home_ml, decimals=2), err_msg

    # Test 2) Cover
    err_msg = "Error in test 1 for test_poisson - incorrect over Cover"
    assert 75 <= np.round(100*cover, decimals=2), err_msg

    
if __name__ == "__main__":
    test_poisson()
    test_nb()