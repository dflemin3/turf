# -*- coding: utf-8 -*-
"""
:py:mod:`utils.py` - Utility functions
--------------------------------------

Utility functions for internal functions and general data processing, e.g.
mapping team names to their standard abbreviations.

@author: David Fleming, 2024
"""


import pandas as pd


__all__ = []


################################################################################
#
# Internal utility functions
#
################################################################################


# Internal dictionary of NFL team names and abbreviations
_nfl_name_conv = {'Tampa Bay Buccaneers' : 'TB',
                  'Atlanta Falcons' : 'ATL',
                  'Buffalo Bills' : 'BUF',
                  'Carolina Panthers' : 'CAR',
                  'Cincinnati Bengals' : 'CIN',
                  'Indianapolis Colts' : 'IND',
                  'Washington Football Team' : 'WAS',
                  'Washington Redskins' : 'WAS',
                  'Washington Commanders' : 'WAS',
                  'Tennessee Titans' : 'TEN',
                  'Houston Oilers' : 'TEN',
                  'Houston Texans' : 'HST',
                  'Detroit Lions' : 'DET',
                  'New Orleans Saints' : 'NO',
                  'New England Patriots' : 'NE',
                  'Kansas City Chiefs' : 'KC',
                  'New York Giants' : 'NYG',
                  'Los Angeles Rams' : 'LAR',
                  'Saint Louis Rams' : 'LAR',
                  'St. Louis Rams' : 'LAR',
                  'Las Vegas Raiders' : 'LV',
                  'Oakland Raiders' : 'LV',
                  'New York Jets' : 'NYJ',
                  'Cleveland Browns' : 'CLV',
                  'Jacksonville Jaguars' : 'JAX',
                  'Miami Dolphins' : 'MIA',
                  'Chicago Bears' : 'CHI',
                  'Philadelphia Eagles' : 'PHI',
                  'Pittsburgh Steelers' : 'PIT',
                  'Arizona Cardinals' : 'ARZ',
                  'San Diego Chargers' : 'LAC',
                  'Los Angeles Chargers' : 'LAC',
                  'Seattle Seahawks' : 'SEA',
                  'Baltimore Ravens' : 'BLT',
                  'Green Bay Packers' : 'GB',
                  'Denver Broncos' : 'DEN',
                  'Minnesota Vikings' : 'MIN',
                  'San Francisco 49ers' : 'SF',
                  'Dallas Cowboys' : 'DAL'}


# Internal dictionary of NHL team names and abbreviations
_nhl_name_conv = {'Arizona Coyotes' : 'ARI',
                   'Anaheim Ducks' : 'ANA',
                   'Boston Bruins' : 'BOS',
                   'Buffalo Sabres' : 'BUF',
                   'Carolina Hurricanes' : 'CAR',
                   'Columbus Blue Jackets' : 'CBJ',
                   'Calgary Flames' : 'CGY',
                   'Chicago Black Hawks' : 'CHI',
                   'Chicago Blackhawks' : 'CHI',
                   'Colorado Avalanche' : 'COL',
                   'Dallas Stars' : 'DAL',
                   'Detroit Red Wings' : 'DET',
                   'Edmonton Oilers' : 'EDM',
                   'Florida Panthers' : 'FLA',
                   'Los Angeles Kings' : 'LAK',
                   'Minnesota Wild' : 'MIN',
                   'Montreal Canadiens' : 'MTL',
                   'New Jersey Devils' : 'NJD',
                   'Nashville Predators' : 'NSH',
                   'New York Islanders' : 'NYI',
                   'New York Rangers' : 'NYR',
                   'Ottawa Senators' : 'OTT',
                   'Philadelphia Flyers' : 'PHI',
                   'Phoenix Coyotes' : 'ARI',
                   'Pittsburgh Penguins' : 'PIT',
                   'Seattle Kraken' : 'SEA',
                   'San Jose Sharks' : 'SJS',
                   'St. Louis Blues' : 'STL',
                   'Saint Louis Blues' : 'STL',
                   'Tampa Bay Lightning' : 'TBL',
                   'Toronto Maple Leafs' : 'TOR',
                   'Vancouver Canucks' : 'VAN',
                   'Vegas Golden Knights' : 'VGK',
                   'Las Vegas Golden Knights' : 'VGK',
                   'Winnipeg Jets' : 'WPG',
                   'Washington Capitals' : 'WSH'}


# NFL hex colors
_hex_color_nfl = {"BLT_0" : "#241773",
                  "BLT_1" : "#000000",
                  "CIN_0" : "#FB4F14",
                  "CIN_1" : "#000000",
                  "CLV_0" : "#311D00",
                  "CLV_1" : "#FF3C00",
                  "PIT_0" : "#FFB612",
                  "PIT_1" : "#101820",
                  "BUF_0" : "#00338D",
                  "BUF_1" : "#C60C30",
                  "MIA_0" : "#008E97",
                  "MIA_1" : "#FC4C02",
                  "NE_0" : "#002244",
                  "NE_1" : "#C60C30",
                  "NYJ_0" : "#125740",
                  "NYJ_1" : "#FFFFFF",
                  "HST_0" : "#03202F",
                  "HST_1" : "#A71930",
                  "IND_0" : "#002C5F",
                  "IND_1" : "#A2AAAD",
                  "JAX_0" : "#006778",
                  "JAX_1" : "#D7A22A",
                  "TEN_0" : "#0C2340",
                  "TEN_1" : "#C8102E",
                  "DEN_0" : "#FB4F14",
                  "DEN_1" : "#002244",
                  "KC_0" : "#E31837",
                  "KC_1" : "#FFB81C",
                  "LV_0" : "#000000",
                  "LV_1" : "#A5ACAF",
                  "LAC_0" : "#0080C6",
                  "LAC_1" : "#FFC20E",
                  "CHI_0" : "#0B162A",
                  "CHI_1" : "#C83803",
                  "DET_0" : "#0076B6",
                  "DET_1" : "#B0B7BC",
                  "GB_0" : "#203731",
                  "GB_1" : "#FFB612",
                  "MIN_0" : "#4F2683",
                  "MIN_1" : "#FFC62F",
                  "DAL_0" : "#003594",
                  "DAL_1" : "#869397",
                  "NYG_0" : "#0B2265",
                  "NYG_1" : "#A71930",
                  "PHI_0" : "#004C54",
                  "PHI_1" : "#A5ACAF",
                  "WAS_0" : "#5A1414",
                  "WAS_1" : "#FFB612",
                  "ATL_0" : "#A71930",
                  "ATL_1" : "#000000",
                  "CAR_0" : "#0085CA",
                  "CAR_1" : "#101820",
                  "NO_0" : "#D3BC8D",
                  "NO_1" : "#101820",
                  "TB_0" : "#D50A0A",
                  "TB_1" : "#34302B",
                  "ARZ_0" : "#97233F",
                  "ARZ_1" : "#000000",
                  "LAR_0" : "#003594",
                  "LAR_1" : "#FFA300",
                  "SF_0" : "#AA0000",
                  "SF_1" : "#B3995D",
                  "SEA_0" : "#002244",
                  "SEA_1" : "#69BE28"}


# NHL hex colors
_hex_color_nhl = {"ANA_0" : "#89734C",
                  "ANA_1" : "#FC4C02",
                  "ARI_0" : "#6F263D",
                  "ARI_1" : "#DDCBA4",
                  "BOS_0" : "#000000",
                  "BOS_1" : "#FFD100",
                  "BUF_0" : "#003087",
                  "BUF_1" : "#FFB81C",
                  "CAR_0" : "#C8102E",
                  "CAR_1" : "#000000",
                  "CGY_0" : "#C8102E",
                  "CGY_1" : "#F1BE48",
                  "CHI_0" : "#C8102E",
                  "CHI_1" : "#000000",
                  "COL_0" : "#6F263D",
                  "COL_1" : "#236192",
                  "CBJ_0" : "#041E42",
                  "CBJ_1" : "#C8102E",
                  "DAL_0" : "#00843D",
                  "DAL_1" : "#A2AAAD",
                  "DET_0" : "#C8102E",
                  "DET_1" : "#FFFFFF",
                  "EDM_0" : "#CF4520",
                  "EDM_1" : "#00205B",
                  "FLA_0" : "#C8102E",
                  "FLA_1" : "#B9975B",
                  "LAK_0" : "#000000",
                  "LAK_1" : "#A2AAAD",
                  "MIN_0" : "#154734",
                  "MIN_1" : "#DDCBA4",
                  "MTL_0" : "#A6192E",
                  "MTL_1" : "#FFFFFF",
                  "NSH_0" : "#FFB81C",
                  "NSH_1" : "#041E42",
                  "NJD_0" : "#C8102E",
                  "NJD_1" : "#FFFFFF",
                  "NYI_0" : "#FC4C02",
                  "NYI_1" : "#003087",
                  "NYR_0" : "#0033A0",
                  "NYR_1" : "#C8102E",
                  "OTT_0" : "#000000",
                  "OTT_1" : "#C8102E",
                  "PHI_0" : "#CF4520",
                  "PHI_1" : "#000000",
                  "PIT_0" : "#000000",
                  "PIT_1" : "#FFB81C",
                  "SJS_0" : "#006272",
                  "SJS_1" : "#E57200",
                  "SEA_0" : "#051C2C",
                  "SEA_1" : "#9CDBD9",
                  "STL_0" : "#003087",
                  "STL_1" : "#FFFFFF",
                  "TBL_0" : "#00205B",
                  "TBL_1" : "#FFFFFF",
                  "TOR_0" : "#00205B",
                  "TOR_1" : "#FFFFFF",
                  "VAN_0" : "#00205B",
                  "VAN_1" : "#FFFFFF",
                  "VGK_0" : "#C69214",
                  "VGK_1" : "#333F48",
                  "WSH_0" : "#C8102E",
                  "WSH_1" : "#041E42",
                  "WPG_0" : "#041E42",
                  "WPG_1" : "#004C97"}


def __nfl_home_away(x : pd.Series) -> str | int | str | int | bool:
    """
    Internal utility function to determine which team is the home/away team
    from a raw pull_season df and create team and score mappings accordingly
    for NFL games
    """

    # Winning team was away
    if x["away_indicator"] == "@":
        away_team = x["Winner/tie"]
        away_pts = x["PtsW"]

        home_team = x["Loser/tie"]
        home_pts = x["PtsL"]
    else:
        home_team = x["Winner/tie"]
        home_pts = x["PtsW"]

        away_team = x["Loser/tie"]
        away_pts = x["PtsL"]

    # Did the game end in a tie?
    if x["PtsW"] == x["PtsL"]:
        tie = True
    else:
        tie = False

    return away_team, away_pts, home_team, home_pts, tie


def _outcome(home_pts : int, away_pts : int) -> bool | bool:
    """
    Simple function for determining the game outcome given the points scored
    by both the home and away team

    Parameters
    ----------
    home_pts : int
        Points scored by home team
    away_pts : int
        Points scored by away team

    Returns
    -------
    home_win : bool
        Whether or not home team wins
    tie : bool
        Whether or not game ends in a tie
    """

    if home_pts > away_pts:
        home_win = True
        tie = False
    elif home_pts < away_pts:
        home_win = False
        tie = False
    else:
        home_win = False
        tie = True

    return home_win, tie


def check_model_inference(model : object) -> bool:
    """
    Check if model specified by model has been inferred by checking if class
    variables ending with "_" have been set, kind of like sklearn. These parameters
    are only set in the run_inference method when it is ran.

    Parameters
    ----------
    model : inference.GenericModel

    Returns
    -------
    ret : bool
        True if model.run_inference(...) has been successfully ran, False otherwise
    """

    if [v for v in vars(model) if v.endswith("_") and not v.startswith("__")]:
        return True
    else:
        return False
