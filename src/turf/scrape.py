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

    def __init__(self, year : int=2022, path : str=None,
                 start_date : str=None, end_date : str=None) -> None: 
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
        start_date : str (optional)
            Date str in %Y-%m-%d format where game results equal or prior to 
            start_date are set to np.nan. Defaults to None.
        end_date : str (optional)
            Date str in %Y-%m-%d format where game results equal or after to 
            end_date are set to np.nan. Defaults to None.
        """

        # Cache year
        self.year = year

        # Season data
        self.raw_season_df = None

        # Cache path
        self.path = path

        # Start and end dates (convert to datetime if possible)
        self.start_date = pd.to_datetime(start_date,
                                         format='%Y-%m-%d') if start_date is not None else start_date
        self.end_date = pd.to_datetime(end_date,
                                      format='%Y-%m-%d') if end_date is not None else end_date
        

    def __repr__(self) -> str:
        """
        String output for print(Season)
        """

        # Information on season year
        ret = f"{self.year}-{self.year+1} season data"
        
        # And when we want data to start (and end)
        if self.start_date is not None:
            ret += f" beginning {self.start_date}"
        if self.end_date is not None:
            ret += f" and ending {self.end_date}"

        return ret

    
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

    def __init__(self, year : int=2022, path : str=None,
                 start_date : str=None, end_date : str=None) -> None:  
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
        path : str (optional)
            Path to pre-cached data to build Season object. If this is provided,
            this file is loaded and used instead of scraping the data.
        start_date : str (optional)
            Date str in %Y-%m-%d format where game results equal or prior to 
            start_date are set to np.nan. Defaults to None.
        end_date : str (optional)
            Date str in %Y-%m-%d format where game results equal or after to 
            end_date are set to np.nan. Defaults to None.
        """

        # Init _GenericModel super (builds model and does everything else)
        super().__init__(year=year, path=path, start_date=start_date,
                         end_date=end_date)

       # Load data from local path or copy df
        if self.path is not None:
            if isinstance(self.path, str):
                self.raw_season_df = pd.read_csv(self.path)
            else:
                raise ValueError("Pass path to csv")
        # Pull raw season data from pro-football-reference.com
        else:
            self.raw_season_df = pull_nfl_full_season_games_raw(year=self.year)

        # Parse dates
        self.raw_season_df['date'] = self.raw_season_df['date'].map(lambda x : pd.to_datetime(x, 
                                                                                              format='%Y-%m-%d'))

        # Convert Week column into string to accomodate playoffs as needed
        self.raw_season_df["week"] = self.raw_season_df["week"].astype(str)

        # Apply a start or end date filter?
        if self.start_date is not None:
            # Set game results to NaN prior to, or equal to, start_date
            mask = (self.raw_season_df['date'] <=  self.start_date)
            self.raw_season_df.loc[mask, 'home_pts'] = np.nan
            self.raw_season_df.loc[mask, 'away_pts'] = np.nan
            self.raw_season_df.loc[mask, 'tie'] = np.nan
        
        if self.end_date is not None:
            # Set game results to NaN after, or equal to, end_date
            mask = (self.raw_season_df['date'] >=  self.end_date)
            self.raw_season_df.loc[mask, 'home_pts'] = np.nan
            self.raw_season_df.loc[mask, 'away_pts'] = np.nan
            self.raw_season_df.loc[mask, 'tie'] = np.nan

        # First save the full season schedule of released games into a separate df
        self.full_schedule = self.raw_season_df[["date", "week", "home_team", "away_team"]].copy()

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

    def __init__(self, year : int=2022, path : str=None,
                 start_date : str=None, end_date : str=None,
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
        path : str (optional)
            Path to pre-cached data to build Season object. If this is provided,
            this file is loaded and used instead of scraping the data.
        start_date : str (optional)
            Date str in %Y-%m-%d format where game results equal or prior to 
            start_date are set to np.nan. Defaults to None.
        end_date : str (optional)
            Date str in %Y-%m-%d format where game results equal or after to 
            end_date are set to np.nan. Defaults to None.
        regulation_adjustment : bool (optional)
            Whether or not to adjust scores to reflect regulation outcomes, that is,
            if a game went to OT and team x won 5-4, the scores would be updated to
            4-4. This is useful if you want your model to purely reflect regulation outcomes.
            Defaults to True
        """

        # Init _GenericModel super (builds model and does everything else)
        super().__init__(year=year, path=path, start_date=start_date,
                         end_date=end_date)

       # Load data from local path
        if self.path is not None:
            if isinstance(self.path, str):
                self.raw_season_df = pd.read_csv(self.path)
            else:
                raise ValueError("Pass path to csv")
        # Pull raw season data from hockey-reference.com
        else:
            self.raw_season_df = pull_nhl_full_season_games_raw(year=self.year)

        # Parse dates
        self.raw_season_df['date'] = self.raw_season_df['date'].map(lambda x : pd.to_datetime(x, 
                                                                                              format='%Y-%m-%d'))

        # Apply a start or end date filter?
        if self.start_date is not None:
            # Set game results to NaN prior to, or equal to, start_date
            mask = (self.raw_season_df['date'] <=  self.start_date)
            self.raw_season_df.loc[mask, 'home_pts'] = np.nan
            self.raw_season_df.loc[mask, 'away_pts'] = np.nan
            self.raw_season_df.loc[mask, 'ot_indicator'] = np.nan
        
        if self.end_date is not None:
            # Set game results to NaN after, or equal to, end_date
            mask = (self.raw_season_df['date'] >=  self.end_date)
            self.raw_season_df.loc[mask, 'home_pts'] = np.nan
            self.raw_season_df.loc[mask, 'away_pts'] = np.nan
            self.raw_season_df.loc[mask, 'ot_indicator'] = np.nan

        # Adjust score to regulation outcomes?
        if regulation_adjustment:
            
            # Get OT outcomes
            mask = self.raw_season_df['ot_indicator']
            away = self.raw_season_df.loc[self.raw_season_df['ot_indicator'],
                                          'away_pts'].values.squeeze()
            home = self.raw_season_df.loc[self.raw_season_df['ot_indicator'],
                                          'home_pts'].values.squeeze()
            
            # Find and QC regulation score
            score = np.max([np.min([away, home], axis=0), np.ones(len(away))], axis=0)
            self.raw_season_df.loc[mask, ['away_pts', 'home_pts']] = np.vstack([score, score]).T

        # First save the full season schedule of released games into a separate df
        self.full_schedule = self.raw_season_df[['date', 'away_team', 'home_team']].copy()

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
    df = pd.read_html(url, parse_dates=False, attrs={'id': 'games'},
                      header=0, index_col=0)[0]

    # Drop rows that are simply dividers
    df = df[~df['Date'].isin(['Date', 'Playoffs', 'Week'])].copy()

    # Parse dates
    df['Date'] = df['Date'].map(lambda x : pd.to_datetime(x, format='%Y-%m-%d'))

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
                                                                 3 : "home_pts",
                                                                 4 : "tie"})
    # Stack back onto dataset
    df = pd.concat([df, new_cols_df], axis='columns')

    # Drop columns we do not need for data processing or inference
    df = df[["Date", "Week", "home_team", "home_pts", "away_team",
             "away_pts", "tie"]].copy()

    # Set dtype
    df["Week"] = df["Week"].astype(str)

    # Rename columns
    df.rename(columns={'Week' : 'week',
                       'Date' : 'date'},
              inplace=True)

    # Map the names to standard internal abbreviations
    df["away_team"] = df["away_team"].map(ut._nfl_name_conv)
    df["home_team"] = df["home_team"].map(ut._nfl_name_conv)

    # Correct datatypes
    df = df.astype({"date" : object,
                    "week" : object,
                    "home_team" : object,
                    "home_pts" : float,
                    "away_team" : object,
                    "away_pts" : float,
                    "tie" : bool})

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

    # Reset index as date is the index
    df.reset_index(drop=False, inplace=True)

    # Rename some column; drop irrelevant ones
    df.rename(columns={"Unnamed: 6" : "ot_indicator", "Visitor" : "away_team",
                       "G" : "away_pts", "Home" : "home_team", "G.1" : "home_pts",
                       "Date" : "date"},
              inplace=True)
    
    # Restrict to columns we care about
    df = df[["date", "home_team", "home_pts", "away_team",
             "away_pts", "ot_indicator"]].copy()

    # Map the names to standard abbreviations
    df["away_team"] = df["away_team"].map(ut._nhl_name_conv)
    df["home_team"] = df["home_team"].map(ut._nhl_name_conv)

    # Convert OT indicator to bool
    df["ot_indicator"] = df["ot_indicator"].map(lambda x : True if ((x == 'OT') or (x == 'SO)')) else False) 
    
    return df