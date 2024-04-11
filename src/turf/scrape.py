# -*- coding: utf-8 -*-
"""
:py:mod:`scrape.py` - Data scraping functions
---------------------------------------------

Functions for scraping NFL game results from pro-football-reference.com
and NHL results from hockey-reference.com. Please follow their robots.txt
and all scraping restrictions listed on the respective websites.

@author: David Fleming, 2024

"""

import pandas as pd
import numpy as  np
from . import utils as ut


__all__ = []


################################################################################
#
# Data scraping classes
#
################################################################################


class _GenericSeason(object):
    """
    Generic season object containing raw data, the full schedule, (un)played games
    and results
    """

    def __init__(self, year : int=2022, path : str=None) -> None: 
        """
        Season object initialization function that pulls data for the given year
        from some url and automatically processes the data to compute the 
        schedule, games played, etc

        Parameters
        ----------
        self : self
        year : int (optional)
            Year in which season begins. Year = 2022 (Default) corresponds to
            the 2022-2023 season, for example.
        path : str (optional)
            Path to pre-cached data to build Season object. If this is provided,
            this file is loaded and used instead of scraping the data.
        """

        # Cache year
        self.year = year

        # Season data
        self.raw_season_df = None

        # Cache path
        self.path = path
        

    def __repr__(self) -> str:
        """
        String output for print(Season)
        """

        return f"{self.year}-{self.year+1} season data"

    
    def save(self, path : str="season.csv") -> None:
        """
        Save raw season data as a csv named path

        Parameters
        ----------
        path : str
            Defaults to "season.csv" in the cwd
        
        Returns
        -------
        None
        """

        # Cache raw season data as csv
        self.raw_season_df.to_csv(path, index=False, header=True)


class NFLSeason(_GenericSeason):
    """
    NFL season object containing raw data, the full schedule, (un)played games
    and results
    """

    def __init__(self, year : int=2022, 
                 path : Union[str | pd.DataFrame]=None) -> None: 
        """
        Season object initialization function that pulls data for the given year
        from https://www.pro-football-reference.com/years/{year}/games.htm and
        automatically processes the data to compute the schedule, games played, etc

        Parameters
        ----------
        self : self
        year : int (optional)
            NFL season such that year corresponds to the year-year+1 NFL season.
            Defaults to 2022 for the 2022-2023 NFL season
        path : str | pd.DataFrame (optional)
            Path to pre-cached data to build Season object. If this is provided,
            this file is loaded and used instead of scraping the data. Or, you 
            could pre-load the data on your own and pass a dataframe.
        """

        # Init _GenericModel super (builds model and does everything else)
        super().__init__(year=year, path=path)

       # Load data from local path or copy df
        if self.path is not None:
            if isinstance(self.path, str):
                self.raw_season_df = pd.read_csv(self.path)
            elif isinstance(self.path, pd.DataFrame):
                self.raw_season_df = self.path.copy()
            else:
                raise ValueError("Pass either pass to csv or dataframe")

        # Pull raw season data from pro-football-reference.com
        else:
            self.raw_season_df = pull_nfl_full_season_games_raw(year=self.year)

        # Convert Week column into string to accomodate playoffs as needed
        self.raw_season_df["Week"] = self.raw_season_df["Week"].astype(str)

        # First save the full season schedule of released games into a separate df
        self.full_schedule = self.raw_season_df[["Date", "Week", "home_team", "away_team"]].copy()

        # Then drop all rows with NaN, that is, those without scores
        self.played_df = self.raw_season_df.dropna(how="any").reset_index(drop=True)

        # Save dataframe of unplayed matchups
        self.unplayed_df = self.raw_season_df[self.raw_season_df["home_pts"].isna()].copy().reset_index(drop=True)


    def __repr__(self) -> str:
        """
        String output for print(Season)
        """

        return f"{self.year}-{self.year+1} NFL season data up to Week {self.week}"



class NHLSeason(_GenericSeason):
    """
    NHL season object containing raw data, the full schedule, (un)played games
    and results
    """

    def __init__(self, year : int=2022, , 
                 path : Union[str | pd.DataFrame]=None
                 regulation_adjustment : bool=True) -> None: 
        """
        Season object initialization function that pulls data for the given year
        from https://www.hockey-reference.com/leagues/NHL_{year+1}_games.html and
        automatically processes the data to compute the schedule, games played, etc

        Parameters
        ----------
        self : self
        year : int (optional)
            NHL season such that year corresponds to the year-year+1 NFL season.
            Defaults to 2022 for the 2022-2023 NHL season
        path : str | pd.DataFrame (optional)
            Path to pre-cached data to build Season object. If this is provided,
            this file is loaded and used instead of scraping the data. Or, you 
            could pre-load the data on your own and pass a dataframe.
        regulation_adjustment : bool (optional)
            Whether or not to adjust scores to reflect regulation outcomes, that is,
            if a game went to OT and team x won 5-4, the scores would be updated to
            4-4. This is useful if you want your model to purely reflect regulation outcomes.
            Defaults to True
        """

        # Init _GenericModel super (builds model and does everything else)
        super().__init__(year=year, path=path)

       # Load data from local path
        if self.path is not None:
            if isinstance(self.path, str):
                self.raw_season_df = pd.read_csv(self.path)
            elif isinstance(self.path, pd.DataFrame):
                self.raw_season_df = self.path.copy()
            else:
                raise ValueError("Pass either pass to csv or dataframe")
        # Pull raw season data from hockey-reference.com
        else:
            self.raw_season_df = pull_nhl_full_season_games_raw(year=self.year)

        # Adjust score to regulation outcomes?
        if regulation_adjustment:
            
            # Get OT outcomes
            mask = (self.raw_season_df['ot_indicator'] == 'OT')
            away = self.raw_season_df.loc[mask, 'away_pts'].values.squeeze()
            home = self.raw_season_df.loc[mask, 'home_pts'].values.squeeze()
            
            # Find and QC regulation score
            score = np.max([np.min([away, home], axis=0), np.ones(len(away))], axis=0)
            self.raw_season_df.loc[mask, ['away_pts', 'home_pts']] = np.vstack([score, score]).T

        # First save the full season schedule of released games into a separate df
        self.full_schedule = self.raw_season_df[['Date', 'away_team', 'home_team']].copy()

        # Then drop all rows with NaN, that is, those without scores
        self.played_df = self.raw_season_df.dropna(subset=['away_pts', 'home_pts'],
                                                   how="any").reset_index(drop=True)

        # Save dataframe of unplayed matchups
        self.unplayed_df = self.raw_season_df[self.raw_season_df["home_pts"].isna()].copy().reset_index(drop=True)


    def __repr__(self) -> str:
        """
        String output for print(Season)
        """

        return f"{self.year}-{self.year+1} NHL season data"


################################################################################
#
# Data scraping functions
#
################################################################################


def pull_nfl_full_season_games_raw(year : int=2022) -> pd.DataFrame:
    """
    Scrape full NFL season game data for the year-year+1 season from
    https://www.pro-football-reference.com/years/{year}/games.htm to get the
    following data for all played and unplayed games:
    "Date", "Week", "home_team", "home_pts", "away_team", "away_pts"

    Parameters
    ----------
    year : int
        Year in which season begins. Year = 2022 (Default) corresponds to
        the 2022-2023 season, for example.

    Returns
    -------
    df : pandas.DataFrame
        processed dataframe of NFL games for the year-year+1 season
    """

    # URL for up-to-date NFL stats
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"

    # Download data
    df = pd.read_html(url, parse_dates=True, attrs={'id': 'games'},
                      header=0, index_col=0)[0]

    # Drop rows that are simply dividers
    df.drop("Week", inplace=True)
    try:
        df.drop("Playoffs", inplace=True)
    except KeyError:
        pass

    # Now reset index so week is a column
    df.reset_index(drop=False, inplace=True)

    # Rename column names
    df.rename(columns={"Unnamed: 5" : "away_indicator", "Unnamed: 7" : "link"},
              inplace=True)

    # Figure out who home and away teams are, how much they scored, respectively
    new_cols_df = df.apply(lambda x: ut.__nfl_home_away(x),
                           axis=1,
                           result_type='expand').rename(columns={0 : "away_team",
                                                                 1 : "away_pts",
                                                                 2 : "home_team",
                                                                 3 : "home_pts"})
    # Stack back onto dataset
    df = pd.concat([df, new_cols_df], axis='columns')

    # Drop columns we do not need for data processing or inference
    df = df[["Date", "Week", "home_team", "home_pts", "away_team",
             "away_pts"]].copy()

    # Set dtype
    df["Week"] = df["Week"].astype(str)

    # Try dropping a dummy divider row
    df = df[df["Date"] != "Playoffs"].copy()

    # Map the names to standard abbreviations
    df["away_team"] = df["away_team"].map(ut._nfl_name_conv)
    df["home_team"] = df["home_team"].map(ut._nfl_name_conv)

    return df


def pull_nhl_full_season_games_raw(year : int=2022) -> pd.DataFrame:
    """
    Scrape full NHL season game data for the year-year+1 season from
    "https://www.hockey-reference.com/leagues/NHL_{year+1}_games.html to get the
    following data for all played and unplayed games:
    "Date", "home_team", "home_pts", "away_team", "away_pts"

    Parameters
    ----------
    year : int
        Year in which season begins. Year = 2022 (Default) corresponds to
        the 2022-2023 season, for example.

    Returns
    -------
    df : pandas.DataFrame
        processed dataframe of NHL games for the year-year+1 season
    """

    # Download NHL schedule data from hockey-reference
    url = f"https://www.hockey-reference.com/leagues/NHL_{year+1}_games.html"
    df = pd.read_html(url, parse_dates=True, attrs={'id': 'games'},
                      header=0, index_col=0)[0]

    # Rename some column; drop irrelevant ones
    df.rename(columns={"Unnamed: 5" : "ot_indicator", "Visitor" : "away_team",
                       "G" : "away_pts", "Home" : "home_team", "G.1" : "home_pts"},
              inplace=True)
    df.drop(columns=["Att.", "LOG", "Notes"], inplace=True)

    # Map the names to standard abbreviations
    df["away_team"] = df["away_team"].map(ut._nhl_name_conv)
    df["home_team"] = df["home_team"].map(ut._nhl_name_conv)

    # Make date a column
    df.reset_index(drop=False, inplace=True)
    
    return df