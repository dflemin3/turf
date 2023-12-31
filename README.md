***turf***

Overview
========

`turf` is a Python library that performs Bayesian hierarchical inference and simulation of
National Football League (NFL) and Nation Hockey League (NHL) games. These models are inspired
by the framework introduced in [Baio and Blangiardo (2010)](https://doi.org/10.1080/02664760802684177).

Tested for Python 3.10 and 3.11.

<p>
<a href="https://github.com/dflemin3/turf">
<img src="https://img.shields.io/badge/GitHub-dflemin3%2Fturf-blue.svg?style=flat"></a>
<a href="https://github.com/dflemin3/turf/blob/master/LICENSE">
<img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat"></a>
</p>

Installation
============

Clone the repository then type
```bash
cd turf
python -m pip install .
```
to install.

Simple example
==============

Below gives a simple example for how to use `turf` to run a hierarchical Bayesian inference on prior NFL season results
to predict the results of the Thursday Night Football game, Denver at Detroit, ran and played on December 16th, 2023.
Here, we assume a Negative Binomial likelihood for the points scored by each team.

```python
import pymc as pm
import numpy as np
from turf import scrape, inference

# Pull season results to-date
season = scrape.NFLSeason(year=2023, week=None)

# Initialize model
model = inference.IndependentNegativeBinomialMixture(season)

# Run inference on 4 cores (1 chain per core)
model.run_inference(tune=2000, draws=1000, target_accept=0.95, chains=4)

# Simulate a game - Denver Broncos at the Detroit Lions
away_team = "DEN"
home_team = "DET"
ou = 48
home_spread = -4.5

# Simulate n game outcomes
home_pts, away_pts, home_win, tie = model.simulate_game(home_team, away_team, n=1000, seed=None)

# Extract results
total = home_pts + away_pts
odds = np.mean(total > ou)
cover = np.mean(home_pts - away_pts > -home_spread)
home_ml = np.mean(home_pts > away_pts)

# Output
print(f"---{away_team} at {home_team}---")
print(f"O/U: {ou} - Over odds : {np.round(100*odds, decimals=2):.2f}%")
print(f"Spread: {home_team} {home_spread} - odds of {home_team} cover : {np.round(100*cover, decimals=2):.2f}%") 
print(f"ML: - odds of {home_team} ML : {np.round(100*home_ml, decimals=2):.2f}%")
print(f"Median outcome: {away_team} {np.median(away_pts):.0f} | {home_team} {np.median(home_pts):.0f}")
print()

# Simulated outcome according to inferred model
# ---DEN at DET---
# O/U: 48 - Over odds : 50.34%
# Spread: DET -4.5 - odds of DET cover : 50.73%
# ML: - odds of DET ML : 61.38%
# Median outcome: DEN 21 | DET 26
```

Analyses
========

Check out the notebook that demonstrate how to characterize teams' offensive and defensive strengths and simulate games:
- [Inference and simulation of NFL games](https://github.com/dflemin3/turf/blob/main/examples/nfl.ipynb)
- [Inference and simulation of NHL games](https://github.com/dflemin3/turf/blob/main/examples/nhl.ipynb)

Check out the following notebook that demonstrates how to calculate a team's strength of schedule (SoS):
- [NFL SoS Estimation](https://github.com/dflemin3/turf/blob/main/examples/nfl_sos.ipynb)
- [NHL SoS Estimation](https://github.com/dflemin3/turf/blob/main/examples/nhl_sos.ipynb)

For more information on the math underpinning the models in `turf` read the [white paper](https://github.com/dflemin3/turf/blob/main/docs/whitepaper.md) 
