"""
:py:mod:`inference.py` - Inference classes, functions
-----------------------------------------------------

Functions and classes for hierarchical inference of NFL and NHL games

@author: David Fleming, 2024

"""


import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from typing import Union
from . import scrape
from . import utils as ut


__all__ = []


################################################################################
#
# Inference classes
#
################################################################################


class _GenericModel(object):
    """
    Abstract class for pymc-based (hierarchical) models of games
    """

    def __init__(self, season : scrape._GenericSeason, path : str=None) -> None:
        """
        Abstract model class _GenericModel object initialization function that
        initializes and defines the typical run_inference sampling function

        Parameters
        ----------
        self : self
        season : turf.scrape._GenericSeason
            Initialized season data that contains data required for inference
        path : str (optional)
            Path to pre-computed trace. Defaults to None, aka, ya need to sample the model

        Returns
        -------
        None
        """

        # Cache season data
        self.season = season

        # Build coords 
        self.build_coords()

        # Build model
        self.build_model()

        # Load trace?
        self.path = path
        if self.path is not None:
            # Load trace
            self.trace_ = az.from_netcdf(self.path)

            # Load number of chains
            self.n_chains_ = self.trace_.posterior.chain.shape[0]

    
    def build_coords(self) -> None:
        """
        Build coords for pymc data management from season results

        Parameters
        ----------
        self

        Returns
        -------
        None
        """

        # Create list of all teams playing in the season and home, away team indicies
        # for observed games, and other dimensions for correlations and groups
        home_idx, teams = pd.factorize(self.season.played_df["home_team"], sort=True)
        away_idx, _ = pd.factorize(self.season.played_df["away_team"], sort=True)
        coords = {"teams" : teams.values, "games" : self.season.played_df.index.values,
                  "att_def" : ['att', 'def'], "groups" : ['bad', 'average', 'good']}

        # Save as internal attributes for reference and inference
        self._played_home_idx = home_idx
        self._played_away_idx = away_idx
        self._played_teams = teams
        self._coords = coords
        self._team_to_idx = {team : idx for idx, team in enumerate(teams)}


    def build_model(self) -> None:
        """
        Build pymc model

        Customize this in subclasses for different (hierarchical) model types

        Parameters
        ----------
        self

        Returns
        -------
        None
        """

        # Implement for subclasses
        raise NotImplementedError()


    def simulate_games(self, home_team : str, away_team : str, n : int=100, 
                       seed : int=None, neutral_site : bool=False,
                       rng : np.random.Generator=None, tie_breaker_pt : bool=False) -> None:
        """
        Simulate an game where away_team plays at home_team and trace
        contains draws from the posterior distribution for model parameters,
        e.g. atts and defs.

        Parameters
        ----------
        home_team : str
            Name of the home team, like STL
        away_team : str
            Name of the away team, like CHI
        n : int (optional)
            Number of games to simulate. Defaults to 100
        seed : int (optional)
            RNG seed. Defaults to 90
        neutral_site : bool (optional)
            Whether game is played at a neutral site or not. Defaults to False
        rng : numpy rng (optional)
            Defaults to None and is initialized internally
        tie_breaker_pt : bool (optional)
            If the game is tied, pick random team with even probability 
            to get 1 OT point. Defaults to False.

        Returns
        -------
        home_pts : int
            number of points scored by the home team
        away_pts : int
            number of points scored by the away team
        home_win : bool
            whether or not the hometeam won
        tie : str
            indicates if the game finished in a tie or not
        """

        # Assert model is fit with new util fn
        assert ut.check_model_inference(self.model), "model must be ran via model.run_inference() prior to simulations"

        # Implement this method for subclasses
        raise NotImplementedError()


    def run_inference(self, draws : int=1000, tune : int=5000, progressbar : bool=True,
                      init : str="jitter+adapt_diag", seed : int=None, chains : int=4,
                      target_accept : float=0.9, cache_loglike : bool=True) -> None:
        """
        Run hierarchical inference for model using pymc

        See https://docs.pymc.io/en/v3/api/inference.html for full parameter documentation

        Results of inference are stored in the self.trace_ attribute

        Parameters
        ----------
        draws : int (optional)
            Number of MCMC draws to take. Defaults to 1000.
        tune : int (optional)
            Number of MCMC tuning steps to take. Defaults to 5000.
        progressbar : bool (optional)
            Whether or not to display the progress bar. Defaults to True.
        init : str (optional)
            Sampler initialization scheme. Defaults to jitter+adapt_diag
        seed : int/list of int (optional)
            RNG seed or seed for each chain. Defaults to None.
        chains : int (optional)
            Number of chains and cores to use to sample posterior. Defaults to 4.
            Do not use more than 4. Caches in model as self.n_chains_
        target_accept : float (optional)
            Target acceptance fraction for HMC. Defaults to 0.9.
        cache_loglike : bool (optional)
            Whether or not to save the loglikelihood in the trace. Defaults to True.

        Returns
        -------
        None

        """

        # QC inputs
        self.n_chains_ = chains

        # Cache loglikelihood?
        if cache_loglike:
            idata_kwargs = {'log_likelihood' : True}
        else:
            idata_kwargs = {}

        # Was the model set?
        err_msg = "Must build self.model before calling run_inference method. See build_model() class method"
        assert self.model is not None, err_msg

        # Run inference
        with self.model:
            self.trace_ = pm.sample(draws=draws, tune=tune, init=init,
                                    progressbar=progressbar, return_inferencedata=True,
                                    random_seed=seed, chains=self.n_chains_, cores=self.n_chains_,
                                    discard_tuned_samples=True, target_accept=target_accept,
                                    idata_kwargs=idata_kwargs)

    
    def sos(self, n : int=100, mode='full', seed : int=None, 
            team_names : list=None) -> Union[np.ndarray, np.ndarray]:
        """
        Simulate season results to estimate a team's strength of schedule 
        as the win percentage of the median team playing the same schedule.

        Parameters
        ----------
        n : int (optional)
            Number of times to simulate each game. Defaults to 100.
        mode : str (optional)
            Estimate SoS for the 'full', 'played', or 'unplayed' games 
            of the season for each team
        seed : int (optional)
            RNG seed. Defaults to None.
        team_names : list of str (optional)
            List of valid team names to simulate. Defaults to None, which
            then simulates the season for all teams in the trace.

        Returns
        -------
        sos : np.ndarray
            Strength of schedule for teams in team_names
        team_names : np.ndarray
            Array of team names aligned with SOS
        """

        # Assert model is fit with new util fn
        assert ut.check_model_inference(self.model), "model must be ran via model.run_inference() prior to simulations"

        # Extract df of all games or remaining games
        if mode == 'full':
            game_df = self.season.full_schedule.copy()
        elif mode == 'played':
            game_df = self.season.played_df.copy()
        elif mode == 'unplayed':
            game_df = self.season.unplayed_df.copy()
        else:
            err_msg = "mode must be one of 'full', 'played', or 'unplayed'. See docstring for more info"
            raise RuntimeError(err_msg)

        # Extract team names from trace
        if team_names is None:
            team_names = self.trace_.posterior.coords['teams'].values
        sos = np.zeros(len(team_names))

        # Simulate winning percentage of median team for given team's schedule
        for ii, team in enumerate(team_names):
            results = []
        
            # Games where team is home team
            mask = (game_df['home_team'] == team)
            home_games = game_df[mask]
            
            # Games where team is away team
            mask = (game_df['away_team'] == team)
            away_games = game_df[mask]
            
            # First play games where team is at home
            for jj in range(len(home_games)):
            
                # Simulate games
                _, _, home_win, _ = self.simulate_game('median', 
                                                       home_games.iloc[jj]['away_team'], n=n, seed=seed)
                results.append(np.mean(home_win))
            
            # Then play games where team is away
            for jj in range(len(away_games)):
            
                # Simulate games
                _, _, home_win, _ = self.simulate_game(away_games.iloc[jj]['home_team'],
                                                       'median', n=n, seed=seed)
                results.append(1 - np.mean(home_win))
        
            # Cache results for team
            sos[ii] = np.mean(results)

        return sos, team_names
    

    def simulation_season(self, n : int=100, mode : str='full',
                          seed : int=None, team_names : list=None) -> dict:
        """
        Simulate {full, played, unplayed} season results for all teams
        in team_names (defaults to all teams)

        Parameters
        ----------
        n : int (optional)
            Number of times to simulate each game. Defaults to 100.
        mode : str (optional)
            Simulate the 'full', 'played', or 'unplayed' games 
            of the season for each team
        seed : int (optional)
            RNG seed. Defaults to None.
        team_names : list of str (optional)
            List of valid team names to simulate. Defaults to None, which
            then simulates the season for all teams in the trace.

        Returns
        -------
        results : dict
            dictionary where keys correspond to team_home and team_away for each
            team. results['team_home'] the stores, for example, the (n_home x n)
            array of home game win results, n for each match.
        """

        # Assert model is fit with new util fn
        assert ut.check_model_inference(self.model), "model must be ran via model.run_inference() prior to simulations"

        # Extract df of all games or remaining games
        if mode == 'full':
            game_df = self.season.full_schedule
        elif mode == 'played':
            game_df = self.season.played_df
        elif mode == 'unplayed':
            game_df = self.season.unplayed_df
        else:
            err_msg = "mode must be one of 'full', 'played', or 'unplayed'. See docstring for more info"
            raise RuntimeError(err_msg)

        # Extract team names from trace?
        if team_names is None:
            team_names = self.trace_.posterior.coords['teams'].values

        # Total, home, and away SOS - nan for all elements at first
        sos = np.zeros(len(team_names),3) * np.nan

        # Simulate winning percentage of median team for given team's schedule
        for ii, team in enumerate(team_names):
        
            # Games where team is home team
            home_mask = (game_df['home_team'] == team)
            home_games = game_df[home_mask]
            n_home = len(home_games)
            
            # Games where team is away team
            away_mask = (game_df['away_team'] == team)
            away_games = game_df[away_mask]
            n_away = len(away_games)
            
            # First play games where team is at home
            if n_home > 0:
                home_results = np.zeros(n_home)
                for jj in range(n_home):
                
                    # Simulate median team playing team at home
                    home_pts, away_pts, home_win, tie = self.simulate_game(home_games.iloc[jj]['home_team'], 
                                                                           home_games.iloc[jj]['away_team'], n=n, seed=seed)
                    home_results[jj] = np.mean(home_win)

                # Estimate home strength of schedule
                sos[ii,1] = np.mean(home_results[jj])
            
            # Then play games where team is away
            if n_away > 0:
                away_results = np.zeros(n_away)
                for jj in range(n_away):
                
                    # Simulate games
                    home_pts, away_pts, home_win, tie = self.simulate_game(away_games.iloc[jj]['home_team'],
                                                           away_games.iloc[jj]['home_team'],
                                                           'median', n=n, seed=seed)
                    away_results[jj] = 1 - np.mean(home_win)

                # Estimate away strength of schedule
                sos[ii,2] = np.mean(away_results[jj])
        
            # Cache results for team - total season, away, home
            sos[ii,0] = sos[ii,1] * (n_home / (n_home + n_away)) + sos[ii,2] * (n_away / (n_home + n_away))
            

        return sos, team_names


    def __repr__(self) -> str:
        """
        String output for printing the model
        """

        if self.model is not None:
            return self.model.__str__()
        else:
            return 'Model has not been initialized'

    
    def save(self, path : str="trace.nc") -> None:
        """
        Save trace to path as a netcdf file

        Parameters
        ----------
        path : str (optional)
            Path to trace for model. Defaults to trace.nc in
            the cwd
        """

        # Assert model is fit with new util fn
        assert ut.check_model_inference(self.model), "model must be ran via model.run_inference() prior to saving"

        # Save to path
        _ = self.trace_.to_netcdf(path)


class IndependentPoisson(_GenericModel):
    """
    Hierarchical generalized linear Poisson model similar to the model disscused
    https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf 
    but assuming log attacking and defensive strengths are uncorrelated but using 
    student t distributions instead of normals
    """

    def __init__(self, season : scrape._GenericSeason, path : str=None) -> None:
        """
        Model object initialization function that takes in and processes
        Season data for use with inference

        Parameters
        ----------
        self : self
        season : turf.scrape.Season
            Initialized season data that contains data required for inference
        path : str (optional)
            Path to pre-computed trace. Defaults to None, aka, ya need to sample the model

        Returns
        -------
        None

        """

        # Init _GenericModel super (builds model and does everything else)
        super().__init__(season=season, path=path)


    def build_model(self) -> None:
        """
        Build pymc-based Sandard model based the initial paper from
        https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf 

        Parameters
        ----------
        self

        Returns
        -------
        None
        """

        # Build pymc model
        with pm.Model(coords=self._coords) as self.model:

            # Constant, observed data
            home_team = pm.Data("home_team",
                                self._played_home_idx,
                                dims="games")
            away_team = pm.Data("away_team",
                                self._played_away_idx,
                                dims="games")
            obs_pts = pm.Data("obs_pts",
                              self.season.played_df[["home_pts", "away_pts"]],
                              dims=("games", "att_def"))

            ### Initialize standard hierarchical model parameters

            # Home effect and typical score intercept term
            home = pm.Normal('home', mu=0.0, sigma=1)
            intercept = pm.Normal('intercept', mu=0.0, sigma=1)

            # Hyperpriors on attack and defense strength standard deviations
            mu_att = pm.Normal("mu_att", mu=0, sigma=1)
            mu_def = pm.Normal("mu_def", mu=0, sigma=1)
            sigma_att = pm.Gamma("sigma_att", alpha=2, beta=0.1)
            sigma_def = pm.Gamma("sigma_def", alpha=2, beta=0.1)

            # Attacking, defensive strength for each team
            atts_star_offset = pm.Normal("atts_star_offset", mu=0, sigma=1, dims="teams")
            atts_star = pm.Deterministic("atts_star", mu_att + atts_star_offset * sigma_att, dims="teams")
            defs_star_offset = pm.Normal("defs_star_offset", mu=0, sigma=1, dims="teams")
            defs_star = pm.Deterministic("defs_star", mu_def + defs_star_offset * sigma_def, dims="teams")

            # Impose "sum-to-zero" constraint
            atts = pm.Deterministic('atts', atts_star - pt.mean(atts_star), dims="teams")
            defs = pm.Deterministic('defs', defs_star - pt.mean(defs_star), dims="teams")

            # Compute theta for the home and away teams (like expected score)
            home_theta = pm.math.exp(intercept + home + atts[home_team] + defs[away_team])
            away_theta = pm.math.exp(intercept + atts[away_team] + defs[home_team])
            theta = pt.stack([home_theta, away_theta]).T

            # Compute home and away point likelihood under log-linear model
            # Recall - model points as a draws from
            # conditionally-independent Poisson distribution: y | theta ~ Poisson(theta)

            # Assume a Poisson likelihood for (uncorrelated) home and away points
            pts = pm.Poisson('pts', mu=theta,                                  # noqa: F841
                             observed=obs_pts, dims=("games", "att_def"))


    def simulate_game(self, home_team : str,
                      away_team : str, n : int=100, seed : int=90, neutral_site : bool=False,
                      rng : np.random.Generator=None,
                      tie_breaker_pt : bool=False) -> Union[int, int, bool, bool]:
        """
        Simulate a game where away_team plays at home_team and trace
        contains draws from the posterior distribution for model parameters,
        e.g. atts and defs.

        Parameters
        ----------
        home_team : str
            Name of the home team, like STL. Can also be "median" where parameters
            are 0 for the median team.
        away_team : str
            Name of the away team, like CHI. Can also be "median" where parameters
            are 0 for the median team.
        n : int (optional)
            Number of games to simulate. Defaults to 100
        seed : int (optional)
            RNG seed. Defaults to 90
        neutral_site : bool (optional)
            Whether game is played at a neutral site or not. Defaults to False
        rng : numpy rng (optional)
            Defaults to None and is initialized internally
        tie_breaker_pt : bool (optional)
            If the game is tied, pick random team with even probability 
            to get 1 OT point. Defaults to False.

        Returns
        -------
        home_pts : int
            number of points scored by the home team
        away_pts : int
            number of points scored by the away team
        home_win : bool
            whether or not the hometeam won
        tie : str
            indicates if the game finished in a tie or not
        """

        # Init rng
        if rng is None:
            rng = np.random.default_rng(seed)

        # Assert model is fit with new util fn
        assert ut.check_model_inference(self.model), "model must be ran via model.run_inference() prior to simulations"

        # Draw random samples with replacement
        inds = rng.choice(self.trace_.posterior.home.shape[-1], size=n, replace=True, shuffle=True)
        chain = rng.integers(self.n_chains_, size=n)

        # Holders
        home = np.zeros(len(inds))
        intercept = np.zeros(len(inds))
        home_att = np.zeros(len(inds))
        home_def = np.zeros(len(inds))
        away_att = np.zeros(len(inds))
        away_def = np.zeros(len(inds))

        # Draw posterior sampls
        for jj, vals in enumerate(zip(inds,chain)):
            
            # Extract indices
            ii, cc = vals

            # Extract parameters
            intercept[jj] = float(self.trace_.posterior.intercept.loc[cc,ii])

            # Played at neutral site?
            if neutral_site:
                home[jj] = 0.0
            else:
                home[jj] = float(self.trace_.posterior.home.loc[cc,ii])
            
            # Extract posterior parameters for team, but allow median team to play
            if home_team == 'median':
                home_att[jj] = 0.0
                home_def[jj] = 0.0
            else:
                home_att[jj] = float(self.trace_.posterior.atts.loc[cc,ii,home_team])
                home_def[jj] = float(self.trace_.posterior.defs.loc[cc,ii,home_team])
            if away_team == 'median':
                away_att[jj] = 0.0
                away_def[jj] = 0.0
            else:
                away_att[jj] = float(self.trace_.posterior.atts.loc[cc,ii,away_team])
                away_def[jj] = float(self.trace_.posterior.defs.loc[cc,ii,away_team])

        # Compute home and away goals using log-linear model, draws for model parameters
        # from posterior distribution. Recall - model points as a draws from
        # conditionally-independent Poisson distribution: y | theta ~ Poisson(theta)
        home_theta = np.exp(home + intercept + home_att + away_def)
        away_theta = np.exp(intercept + away_att + home_def)
        home_pts = rng.poisson(home_theta)
        away_pts = rng.poisson(away_theta)

        # Evaluate and process game results to more standard win, loss, etc nomenclature
        outcomes = np.asarray([[ut._outcome(hpt, apt)] for hpt, apt in zip(home_pts, away_pts)]).squeeze()
        home_win, tie = outcomes[:,0], outcomes[:,1]

        # Assign tiebreaker point?
        for ii in range(len(home_win)):
            if tie[ii] and tie_breaker_pt:
                hw = rng.integers(low=0, high=1, size=1)
                if hw == 1:
                    home_pts[ii] += 1
                else:
                    away_pts[ii] += 1

        return home_pts, away_pts, home_win, tie



class CorrelatedPoisson(IndependentPoisson):
    """
    Hierarchical generalized linear Poisson model similar to the model disccused
    https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf but with correlated 
    log attacking and defensive strength via a multi-variate student t's

    """

    def __init__(self, season : scrape._GenericSeason, path : str=None) -> None:
        """
        Model object initialization function that takes in and processes
        Season data for use with inference

        Parameters
        ----------
        self : self
        season : turf.scrape.Season
            Initialized season data that contains data required for inference
        path : str (optional)
            Path to pre-computed trace. Defaults to None, aka, ya need to sample the model

        Returns
        -------
        None

        """

        # Init StandardModel super (builds model and does everything else)
        super().__init__(season=season, path=path)


    def build_model(self) -> None:
        """
        Build pymc-based Sandard model based the initial paper from
        https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf 
        
        Parameters
        ----------
        self

        Returns
        -------
        None
        """

        # Build pymc model
        with pm.Model(coords=self._coords) as self.model:

            # Constant, observed data
            home_team = pm.Data("home_team",
                                self._played_home_idx,
                                dims="games")
            away_team = pm.Data("away_team",
                                self._played_away_idx,
                                dims="games")
            obs_pts = pm.Data("obs_pts",
                              self.season.played_df[["home_pts", "away_pts"]].values,
                              dims=("games", "att_def"))

            ### Initialize hierarchical model parameters

            # Home effect and typical score intercept term
            home = pm.Normal('home', mu=0.0, sigma=1)
            intercept = pm.Normal('intercept', mu=0.0, sigma=1)

            # Prior standard deviation for att and def terms
            lkj_sd = pm.HalfNormal.dist(shape=2)

            # Prior on correlation matrix
            chol, corr, stds = pm.LKJCholeskyCov("chol_cov", n=2, eta=1, sd_dist=lkj_sd,
                                                 compute_corr=True, store_in_trace=True)
            
            # Attacking, defensive strength for each team modeled as multivariate normal
            atts_defs_star = pm.MvNormal('atts_defs_star', mu=0, chol=chol, dims=("teams", "att_def"))

            # Impose "sum-to-zero" constraint
            atts = pm.Deterministic('atts', atts_defs_star[:,0] - pt.mean(atts_defs_star[:,0]), dims="teams")
            defs = pm.Deterministic('defs', atts_defs_star[:,1] - pt.mean(atts_defs_star[:,1]), dims="teams")

            # Additive model for scores in log space (like log expected score)
            log_home_theta = intercept + home + atts[home_team] + defs[away_team]
            log_away_theta = intercept + atts[away_team] + defs[home_team]
            log_theta = pt.stack([log_home_theta, log_away_theta]).T
            theta = pm.math.exp(log_theta)

            # Compute home and away point likelihood under log-linear model
            # Recall - model points as a draws from
            # conditionally-independent Poisson distribution: y | theta ~ Poisson(theta)

            # Assume a Poisson likelihood for (correlated in log theta space) home and away points
            pts = pm.Poisson('pts', mu=theta,                                              # noqa: F841
                             observed=obs_pts, dims=("games", "att_def")) 


class IndependentNegativeBinomial(_GenericModel):
    """
    Hierarchical generalized linear Poisson model similar to the model disccused
    https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf with the NegativeBinomial
    likelihood

    """

    def __init__(self, season : scrape._GenericSeason, path : str=None) -> None:
        """
        Model object initialization function that takes in and processes
        Season data for use with inference

        Parameters
        ----------
        self : self
        season : turf.scrape._GenericSeason
            Initialized season data that contains data required for inference
        path : str (optional)
            Path to pre-computed trace. Defaults to None, aka, ya need to sample the model

        Returns
        -------
        None

        """

        # Init StandardModel super (builds model and does everything else)
        super().__init__(season=season, path=path)


    def build_model(self) -> None:
        """
        Build pymc-based Sandard model based the initial paper from
        https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf 
        
        Parameters
        ----------
        self

        Returns
        -------
        None
        """

        # Build pymc model
        with pm.Model(coords=self._coords) as self.model:

            # Constant, observed data
            home_team = pm.Data("home_team",
                                self._played_home_idx,
                                dims="games")
            away_team = pm.Data("away_team",
                                self._played_away_idx,
                                dims="games")
            obs_pts = pm.Data("obs_pts",
                              self.season.played_df[["home_pts", "away_pts"]].values,
                              dims=("games", "att_def"))

            ### Initialize standard hierarchical model parameters

            # Home effect and typical score intercept term
            home = pm.Normal('home', mu=0.0, sigma=1)
            intercept = pm.Normal('intercept', mu=0.0, sigma=1)

            # Hyperpriors on attack and defense strength means and standard deviations
            mu_att = pm.Normal("mu_att", mu=0, sigma=1)
            mu_def = pm.Normal("mu_def", mu=0, sigma=1)
            sigma_att = pm.Gamma("sigma_att", alpha=2, beta=0.1)
            sigma_def = pm.Gamma("sigma_def", alpha=2, beta=0.1)

            # Prior for alphas (home and away)
            alpha = pm.Gamma("alpha", alpha=2, beta=0.1, dims="att_def")

            # Attacking, defensive strength for each team
            atts_star_offset = pm.Normal("atts_star_offset", mu=0, sigma=1, dims="teams")
            atts_star = pm.Deterministic("atts_star", mu_att + atts_star_offset * sigma_att, dims="teams")
            defs_star_offset = pm.Normal("defs_star_offset", mu=0, sigma=1, dims="teams")
            defs_star = pm.Deterministic("defs_star", mu_def + defs_star_offset * sigma_def, dims="teams")

            # Impose "sum-to-zero" constraint
            atts = pm.Deterministic('atts', atts_star - pt.mean(atts_star), dims="teams")
            defs = pm.Deterministic('defs', defs_star - pt.mean(defs_star), dims="teams")

            # Compute theta for the home and away teams (like expected score)
            home_theta = pm.math.exp(intercept + home + atts[home_team] + defs[away_team])
            away_theta = pm.math.exp(intercept + atts[away_team] + defs[home_team])
            theta = pt.stack([home_theta, away_theta]).T

            # Compute home and away point likelihood under log-linear model
            # Recall - model points as a draws from
            # conditionally-independent NegativeBinomial distribution: y | theta ~ NB(theta)

            # Assume a Negative Binomial likelihood for (uncorrelated) home and away points
            pts = pm.NegativeBinomial('pts', mu=theta, alpha=alpha,                  # noqa: F841
                                      observed=obs_pts, dims=("games", "att_def")) 

    
    def simulate_game(self, home_team : str,
                      away_team : str, n : int=100, seed : int=90, neutral_site : bool=False,
                      rng : np.random.Generator=None,
                      tie_breaker_pt : bool=False) -> Union[int, int, bool, bool]:
        """
        Simulate an game where away_team plays at home_team and trace
        contains draws from the posterior distribution for model parameters,
        e.g. atts and defs.

        Parameters
        ----------
        home_team : str
            Name of the home team, like STL. Can also be "median" where parameters
            are 0 for the median team.
        away_team : str
            Name of the away team, like CHI. Can also be "median" where parameters
            are 0 for the median team.
        n : int (optional)
            Number of games to simulate. Defaults to 100
        seed : int (optional)
            RNG seed. Defaults to 90
        neutral_site : bool (optional)
            Whether game is played at a neutral site or not. Defaults to False
        rng : numpy rng (optional)
            Defaults to None and is initialized internally
        tie_breaker_pt : bool (optional)
            If the game is tied, pick random team with even probability 
            to get 1 OT point. Defaults to False.

        Returns
        -------
        home_pts : int
            number of points scored by the home team
        away_pts : int
            number of points scored by the away team
        home_win : bool
            whether or not the hometeam won
        tie : str
            indicates if the game finished in a tie or not
        """

        # Init rng
        if rng is None:
            rng = np.random.default_rng(seed)

        # Assert model is fit with new util fn
        assert ut.check_model_inference(self.model), "model must be ran via model.run_inference() prior to simulations"

        # Draw random samples with replacement
        inds = rng.choice(self.trace_.posterior.home.shape[-1], size=n, replace=True, shuffle=True)
        chain = rng.integers(self.n_chains_, size=n)

        # Holders
        home = np.zeros(len(inds))
        intercept = np.zeros(len(inds))
        alpha = np.zeros((len(inds),2))
        home_att = np.zeros(len(inds))
        home_def = np.zeros(len(inds))
        away_att = np.zeros(len(inds))
        away_def = np.zeros(len(inds))

        # Draw posterior sampls
        for jj, vals in enumerate(zip(inds,chain)):
            
            # Extract indices
            ii, cc = vals

            # Extract parameters
            intercept[jj] = float(self.trace_.posterior.intercept.loc[cc,ii])
            alpha[jj,:] = self.trace_.posterior.alpha.loc[cc,ii]

            # Played at neutral site?
            if neutral_site:
                home[jj] = 0.0
            else:
                home[jj] = float(self.trace_.posterior.home.loc[cc,ii])
            
            # Extract posterior parameters for team, but allow median team to play
            if home_team == 'median':
                home_att[jj] = 0.0
                home_def[jj] = 0.0
            else:
                home_att[jj] = float(self.trace_.posterior.atts.loc[cc,ii,home_team])
                home_def[jj] = float(self.trace_.posterior.defs.loc[cc,ii,home_team])
            if away_team == 'median':
                away_att[jj] = 0.0
                away_def[jj] = 0.0
            else:
                away_att[jj] = float(self.trace_.posterior.atts.loc[cc,ii,away_team])
                away_def[jj] = float(self.trace_.posterior.defs.loc[cc,ii,away_team])

            # Is game played at a neutral site?
            if neutral_site:
                home[jj] = 0.0

        # Compute home and away goals using log-linear model, draws for model parameters
        # from posterior distribution. Recall - model points as a draws from
        # conditionally-independent Negative Binomial distribution: y | theta ~ NB(theta)
        home_theta = np.exp(home + intercept + home_att + away_def)
        away_theta = np.exp(intercept + away_att + home_def)
        home_pts = rng.negative_binomial(alpha[:,0], alpha[:,0]/(alpha[:,0]+home_theta))
        away_pts = rng.negative_binomial(alpha[:,1], alpha[:,1]/(alpha[:,1]+away_theta))

        # Evaluate and process game results to more standard win, loss, etc nomenclature
        outcomes = np.asarray([[ut._outcome(hpt, apt)] for hpt, apt in zip(home_pts, away_pts)]).squeeze()
        home_win, tie = outcomes[:,0], outcomes[:,1]

        # Assign tiebreaker point?
        for ii in range(len(home_win)):
            if tie[ii] and tie_breaker_pt:
                hw = rng.integers(low=0, high=1, size=1)
                if hw == 1:
                    home_pts[ii] += 1
                else:
                    away_pts[ii] += 1

        return home_pts, away_pts, home_win, tie

    
class IndependentNegativeBinomialMixture(IndependentNegativeBinomial):
    """
    Hierarchical generalized linear Negative Binomial model similar to the model disscused
    https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf 
    assuming log attacking and defensive strengths are uncorrelated but using 
    normals for bad, average, and good teams with a Dirichlet prior on group membership
    """

    def __init__(self, season : scrape._GenericSeason, path : str=None) -> None:
        """
        Model object initialization function that takes in and processes
        Season data for use with inference

        Parameters
        ----------
        self : self
        season : turf.scrape._GenericSeason
            Initialized season data that contains data required for inference
        path : str (optional)
            Path to pre-computed trace. Defaults to None, aka, ya need to sample the model

        Returns
        -------
        None

        """

        # Init _GenericModel super (builds model and does everything else)
        super().__init__(season=season, path=path)


    def build_model(self) -> None:
        """
        Build pymc-based model based the initial paper from
        https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf, but using
        a Negative Binomial link function and a Dirichlet-prior mixture of
        normals

        Parameters
        ----------
        self

        Returns
        -------
        None
        """

        # Build pymc model
        with pm.Model(coords=self._coords) as self.model:

            # Constant, observed data
            home_team = pm.Data("home_team",
                                self._played_home_idx,
                                dims="games")
            away_team = pm.Data("away_team",
                                self._played_away_idx,
                                dims="games")
            obs_pts = pm.Data("obs_pts",
                              self.season.played_df[["home_pts", "away_pts"]].values,
                              dims=("games", "att_def"))

            ### Initialize standard hierarchical model parameters

            # Home effect, typical score intercept term
            home = pm.Normal('home', mu=0.0, sigma=1)
            intercept = pm.Normal('intercept', mu=0.0, sigma=1)
            
            # Attacking mixture weights
            att_weights_unscaled = pm.Dirichlet('att_weights_unscaled', 
                                                a=np.ones((3, len(self._coords['teams']))),
                                                dims=("groups", "teams"))
            att_weights = pm.Deterministic('att_weights', att_weights_unscaled / pm.math.sum(att_weights_unscaled, axis=0))
            
            # Defense mixture weights
            def_weights_unscaled = pm.Dirichlet('def_weights_unscaled',
                                                a=np.ones((3, len(self._coords['teams']))),
                                                dims=("groups", "teams"))
            def_weights = pm.Deterministic('def_weights', def_weights_unscaled / pm.math.sum(def_weights_unscaled, axis=0))

            # Hyperpriors on attack and defense strength standard deviations
            mu_att = pm.Normal("mu_att", mu=[-1, 0, 1], sigma=1, dims="groups")
            mu_def = pm.Normal("mu_def", mu=[1, 0, -1], sigma=1, dims="groups")
            sigma_att = pm.Gamma("sigma_att", alpha=2, beta=0.1, dims="groups")
            sigma_def = pm.Gamma("sigma_def", alpha=2, beta=0.1, dims="groups")

            # Prior for alphas (home and away)
            alpha = pm.Gamma("alpha", alpha=2, beta=0.1, dims="att_def")

            # Attacking strength for each team by group
            atts_star_offset = pm.Normal("att_star_offset", mu=0, sigma=1, dims=("groups", "teams"))
            atts_star_bad = pm.Deterministic("atts_star_bad", mu_att[0] + atts_star_offset[0,:] * sigma_att[0],
                                             dims="teams")
            atts_star_avg = pm.Deterministic("atts_star_avg", mu_att[1] + atts_star_offset[1,:] * sigma_att[1],
                                             dims="teams")
            atts_star_good = pm.Deterministic("atts_star_good", mu_att[2] + atts_star_offset[2,:] * sigma_att[2],
                                              dims="teams")
            
            # Defensive strength for each team by group
            defs_star_offset = pm.Normal("def_star_offset", mu=0, sigma=1, dims=("groups", "teams"))
            defs_star_bad = pm.Deterministic("defs_star_bad", mu_def[0] + defs_star_offset[0,:] * sigma_def[0],
                                             dims="teams")
            defs_star_avg = pm.Deterministic("defs_star_avg", mu_def[1] + defs_star_offset[1,:] * sigma_def[1],
                                             dims="teams")
            defs_star_good = pm.Deterministic("defs_star_good", mu_def[2] + defs_star_offset[2,:] * sigma_def[2],
                                              dims="teams")

            # Atts, defs and mixture of bad, avg, and good components
            atts_star = pm.Deterministic("atts_star", att_weights[0,:] * atts_star_bad + 
                                         att_weights[1,:] * atts_star_avg + 
                                         att_weights[2,:] * atts_star_good, dims="teams")
            defs_star = pm.Deterministic("defs_star", def_weights[0,:] * defs_star_bad + 
                                         def_weights[1,:] * defs_star_avg + 
                                         def_weights[2,:] * defs_star_good, dims="teams")

            # Impose "sum-to-zero" constraint
            atts = pm.Deterministic('atts', atts_star - pt.mean(atts_star), dims="teams")
            defs = pm.Deterministic('defs', defs_star - pt.mean(defs_star), dims="teams")

            # Compute theta for the home and away teams (like expected score)
            home_log_theta = intercept + home + atts[home_team] + defs[away_team]
            away_log_theta = intercept + atts[away_team] + defs[home_team]
            log_theta = pt.stack([home_log_theta, away_log_theta]).T
            theta = pm.math.exp(log_theta)

            # Compute home and away point likelihood under log-linear model
            # Recall - model points as a draws from
            # conditionally-independent NegativeBinomial distribution: y | theta ~ NB(theta)

            # Assume a Negative Binomial likelihood for (uncorrelated) home and away points
            pts = pm.NegativeBinomial('pts', mu=theta, alpha=alpha,                        # noqa: F841
                                      observed=obs_pts, dims=("games", "att_def")) 