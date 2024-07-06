# turf

### Overview
---

`turf` is a Python library that performs Bayesian hierarchical inference and simulation of
National Football League (NFL) and Nation Hockey League (NHL) games. These models are inspired
by the framework introduced in [Baio and Blangiardo (2010)](https://doi.org/10.1080/02664760802684177).

Tested for Python 3.10, 3.11, and 3.12.

<p>
<a href="https://github.com/dflemin3/turf">
<img src="https://img.shields.io/badge/GitHub-dflemin3%2Fturf-blue.svg?style=flat"></a>
<a href="https://github.com/dflemin3/turf/blob/master/LICENSE">
<img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat"></a>
</p>

### Installation
---

Clone the repository then type
```bash
cd turf
pip install .
```
to install.

If you want run the example notebooks, unit tests, and more run
```bash
pip install .[opp]
```
to install with visualization support from `matplotlib`, `arviz`, `seaborn`. `pytest`, and
`coverage` for unit tests and code coverage, and `ruff` for linting.
If that does not work, which, for example, can happen if you use a zsh shell, try
```bash
pip install '.[opp]'
```

If you lack admin permissions, add the `-e` flag: `pip install -e .`. If that does not work, 
see the official `pip` documentation for additional options for installing locally or in edit/development mode.

### Example
---

Here is an example for how to use `turf` to run a hierarchical Bayesian inference on NFL season results
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

### Analyses
---

Check out the notebook that demonstrate how to characterize teams' offensive and defensive strengths and simulate games:
- [Inference and simulation of NFL games](https://github.com/dflemin3/turf/blob/main/examples/nfl.ipynb)
- [Inference and simulation of NHL games](https://github.com/dflemin3/turf/blob/main/examples/nhl.ipynb)

Check out the following notebook that demonstrates how to calculate a team's strength of schedule (SoS):
- [NFL SoS Estimation](https://github.com/dflemin3/turf/blob/main/examples/nfl_sos.ipynb)
- [NHL SoS Estimation](https://github.com/dflemin3/turf/blob/main/examples/nhl_sos.ipynb)

For more information on the math underpinning the models in `turf` read the 
[white paper](https://github.com/dflemin3/turf/blob/main/docs/whitepaper.md) 

### Tests and coverage

---

This repository uses `GitHub Actions` to automate linting using [`ruff`](https://github.com/astral-sh/ruff), unit tests with
[`pytest`](https://pytest.org), and code coverage calculation with [`coverage`](https://coverage.readthedocs.io/). We also use
`dependabot` to mitigate potential security vulnerabilities. These actions generally trigger for all pull requests and changes to
main. 

To run linting locally using `ruff`, run

```bash
ruff check
```

For unit tests and coverage, run

```bash
coverage run -m pytest -v -s
coverage report -m
```

in the main directory, `turf` (not `src/turf`) after installing the package with additional options (see above).

### Contributing

---

Making contributions to this code base, like documentation, unit tests, enhancements, etc... are more than welcome! Please 
feel free to [fork](https://github.com/dflemin3/turf/fork) the repository, commit some code and then open a 
[pull request](https://github.com/dflemin3/turf/pulls) to see if we can merge the code. Open an
[issue](https://github.com/dflemin3/turf/issues/new/choose) to identify a bug, request a new feature, or start a relevant discussion!