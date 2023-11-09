***turf***

Overview
========

`turf` is a Python library that performs Bayesian hierarchical inference and simulation of NFL games

Runs for Python 3.9+

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

Below gives a simple example for how to use `turf` to run a hierarchical Bayesian inference on NFL season results
for the Thursday Night Football game, Carolina at Chicago, on November 9th, 2023.

```python
import pymc as pm
import numpy as np
from turf import scrape, inference

# Pull season results to-date
season = scrape.Season(year=2023, week=None)

# Initialize model where points
model = inference.CorrelatedPoisson(season)

# Run inference on 4 cores (1 chain per core)
model.run_inference(tune=1000, draws=1000, target_accept=0.9, chains=4)

# Simulate a game - Buffalo Bills at the Cincinnati Bengals
away_team = "CAR"
home_team = "CHI"
ou = 38
home_spread = -3.5

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

# ---CAR at CHI---
# O/U: 38 - Over odds : 82.64%
# Spread: CHI -3.5 - odds of CHI cover : 56.10%
# ML: - odds of CHI ML : 71.51%
# Median outcome: CAR 20 | CHI 25
```

Analyses
========

Check out the notebook that demonstrate how to characterize teams' offensive and defensive strengths and simulate games:
- [Example and simulation](https://github.com/dflemin3/turf/blob/main/examples/example.ipynb)

Check out the following notebook that demonstrates how to calculate a team's strength of schedule (SoS):
- [SoS Estimation](https://github.com/dflemin3/turf/blob/main/examples/sos.ipynb)
