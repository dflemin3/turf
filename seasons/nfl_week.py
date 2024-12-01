# -*- coding: utf-8 -*-
"""
Script for simulating the outcome of a series of NFL games
using a hierarchical Bayesian inference model

@author: David Fleming, 2024

"""

import numpy as np
import pandas as pd
import os
import pymc as pm
import arviz as az
from turf import scrape, utils, inference
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# RNG seed
seed = None

# Number of sims to run per game
n_sims = 25000

# Year, week
year = 2024
week = 13

# Compute metrics and figures?
posterior_metrics = True

# Get today's date
today = datetime.today().strftime('%Y_%m_%d')

# Path to data (make dir for data if it doesn't exist)
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, f'nfl_{year}_week{week}')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

season_path = os.path.join(target_dir, f'nfl_{year}_week{week}_{today}_data.csv')
model_path = os.path.join(target_dir, f'nfl_{year}_week{week}_{today}_trace.nc')
sims_path = os.path.join(target_dir, f'nfl_{year}_week{week}_{today}_sims.csv')

# Pull season results to-date
if not os.path.exists(season_path):
    print(f"Scraping and caching season data for Week {week} year {year}...")
    season = scrape.NFLSeason(year=year)
    season.save(season_path)
else:
    print(f"Loading pre-scraped season data for Week {week} year {year}...")
    season = scrape.NFLSeason(year=year, path=season_path)

# Sample model if we have not already
if not os.path.exists(model_path):

    # Initialize model
    model = inference.IndependentNegativeBinomialMixture(season)
    
    # Run inference on 4 cores (1 chain per core)
    print("Sampling posterior distribution...")
    model.run_inference(tune=2000, draws=1000, target_accept=0.95, chains=4)

    # Cache results
    print(f"Saving model to {model_path}...")
    model.save(model_path)

# Initialize pre-computed model 
print(f"Loading model from {model_path}...")
model = inference.IndependentNegativeBinomialMixture(season, path=model_path)

# Compute posterior metrics?
if posterior_metrics:

    # Compute and cache posterior metrics
    az.summary(model.trace_, group='posterior').to_csv(os.path.join(target_dir, 'posterior_metrics.csv'), 
                                                       index=False, 
                                                       header=True)

    # Sample from posterior predictive distribution
    print("Sampling from posterior predictive distribution...")
    with model.model:
        model.trace_.extend(pm.sample_posterior_predictive(model.trace_))

    ### Plot posterior predictive eCDF
    ax = az.plot_ppc(model.trace_, kind='cumulative', num_pp_samples=1000, group='posterior')
    sns.despine(ax=ax)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Total points")
    ax.set_ylabel("Cumulative probability")
    ax.get_figure().savefig(os.path.join(target_dir, 'pp_ecdf.png'), bbox_inches='tight')
    
    ### Examine full posterior distributions and trace for select parameters
    axes = az.plot_trace(model.trace_, compact=True, var_names=["intercept", "home", "alpha",
                                                                "atts", "defs", "mu_att", "sigma_att",
                                                                "mu_def", "sigma_def"],
                        combined=True)

    for ii in range(axes.shape[0]):
        for jj in range(axes.shape[1]):
            ax = axes[ii,jj]
            sns.despine(ax=ax)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(os.path.join(target_dir, 'posterior.png'), bbox_inches='tight')
    plt.clf()

    ### Extract some data for plots
    
    # Team names, medians, CIs, and colors
    team_names = model.trace_.posterior.coords['teams'].values

    # Calculate median, 68% CI for atts, defs for each team
    med_atts = model.trace_.posterior["atts"].median(axis=1).median(axis=0).values
    med_defs = model.trace_.posterior["defs"].median(axis=1).median(axis=0).values

    # Calculate median, 68% CI for atts, defs for each team, convert to numpy arrays
    defs_CI = az.hdi(model.trace_, var_names=["defs"], hdi_prob=0.68)
    defs_CI = defs_CI.to_array().values[0]

    atts_CI = az.hdi(model.trace_, var_names=["atts"], hdi_prob=0.68)
    atts_CI = atts_CI.to_array().values[0]

    # Get primary and secondary colors for pretty plots
    pri_colors = list(map(lambda x : utils._hex_color_nfl[f"{x}_0"], team_names))
    sec_colors = list(map(lambda x : utils._hex_color_nfl[f"{x}_1"], team_names))

    ### Plot ordered attacking strength
    fig, ax = plt.subplots(figsize=(10,4))

    # Order values by worst to best attacking
    inds = np.argsort(med_atts)

    x = np.arange(len(med_atts))
    for ii in range(len(x)):
        ax.errorbar(x[ii], med_atts[inds[ii]], 
                    yerr=np.asarray([med_atts[inds[ii]] - atts_CI[inds[ii],0], atts_CI[inds[ii],1] - med_atts[inds[ii]]]).reshape(2,1),
                    fmt='o', color=pri_colors[inds[ii]])

    ax.axhline(0, lw=2, ls="--", color="k", zorder=0)
    ax.set_xlabel('')
    ax.set_ylabel('Posterior offensive strength\n' + r'More positive is better$\longrightarrow$')
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(model._coords["teams"][inds], rotation=45)
    sns.despine(ax=ax)
    fig.savefig(os.path.join(target_dir, 'post_att.png'), bbox_inches='tight')
    plt.clf()

    ### Plot ordered attacking strength
    fig, ax = plt.subplots(figsize=(10,4))

    # Order values by worst to best attacking
    inds = np.argsort(med_defs)[::-1]

    x = np.arange(len(med_defs))
    for ii in range(len(x)):
        ax.errorbar(x[ii], med_defs[inds[ii]], 
                    yerr=np.asarray([med_defs[inds[ii]] - defs_CI[inds[ii],0], defs_CI[inds[ii],1] - med_defs[inds[ii]]]).reshape(2,1),
                    fmt='o', color=pri_colors[inds[ii]])

    ax.axhline(0, lw=2, ls="--", color="k", zorder=0)
    ax.set_xlabel('Team')
    ax.set_ylabel('Posterior defensive strength\n' + r'$\longleftarrow$More negative is better')
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(model._coords["teams"][inds], rotation=45)
    sns.despine(ax=ax)
    fig.savefig(os.path.join(target_dir, 'post_def.png'), bbox_inches='tight')
    plt.clf()

### Simulate select games with odds (collected at time of running script)
away_teams = ["PIT", "ARZ", "LAC", "SEA", "TEN", "HST", "IND", "LAR", "TB", "PHI", "SF"]
home_teams = ["CIN", "MIN", "ATL", "NYJ", "WAS", "JAX", "NE", "NO", "CAR", "BLT", "BUF"]
ous = [46, 45, 47, 42, 44.5, 45, 42.5, 49, 46.5, 51, 44.5]
ous_vegas = [-110, -110, -110, -110, -110, -110, -110, -110, -110, -110, -110]
home_spreads = [-3, -3.5, 1, -1.5, -6, 3.5, 2.5, 2.5, 6.5, -3, -6.5]
home_spreads_vegas = [-110, -105, -105, -110, -105, -115, -110, -105, -110, -105, -110]
home_ml_vegas = [-160, -170, 100, -125, 260, 143, 118, 125, 235, -165, -285]

# Holders
totals = []
overs = []
covers = []
home_mls = []
home_pts_mean = []
away_pts_mean = []

for home_team, away_team, ou, hs, ii, in zip(home_teams, away_teams, ous, home_spreads, np.arange(len(home_teams))):

    # Simulate n game outcomes
    home_pts, away_pts, home_win, tie = model.simulate_game(home_team, away_team, n=n_sims, seed=seed,
                                                            neutral_site=False)
    
    total = home_pts + away_pts
    odds = np.mean(total > ou)
    cover = np.mean(home_pts - away_pts > -hs)
    home_ml = np.mean(home_pts > away_pts)
    
    # Print, then cache, results
    print(f"---{away_team} at {home_team}---")
    print(f"O/U: {ou} - Over odds : {np.round(100*odds, decimals=2):.1f}%")
    print(f"Spread: {home_team} {home_spreads[ii]} - odds of {home_team} cover : {np.round(100*cover, decimals=2):.1f}%") 
    print(f"ML: - odds of {home_team} ML : {np.round(100*home_ml, decimals=2):.1f}%")
    print(f"Median scores: {away_team} {np.median(away_pts):.0f} | {home_team} {np.median(home_pts):.0f}")
    print()

    totals.append(np.mean(total))
    overs.append(odds)
    covers.append(cover)
    home_mls.append(home_ml)
    home_pts_mean.append(np.mean(home_pts))
    away_pts_mean.append(np.mean(away_pts))

# Save result as df
sims = pd.DataFrame.from_dict({'home_team' : home_teams,
                               'away_team' : away_teams,
                               'n_sims' : [n_sims for _ in range(len(home_teams))],
                               'over_under' : ous,
                               'over_under_vegas_odds' : ous_vegas,
                               'home_spread' : home_spreads,
                               'home_spread_vegas_odds' : home_spreads_vegas,
                               'home_ml_vegas_odds' : home_ml_vegas,
                               'mean_total_pts' : totals,
                               'mean_home_pts' : home_pts_mean,
                               'mean_away_pts' : away_pts_mean,
                               'prob_over' : overs,
                               'prob_home_cover' : covers,
                               'prob_home_ml' : home_mls})
sims.to_csv(sims_path, index=False, header=True)
