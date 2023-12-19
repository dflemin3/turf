# The math behind `turf` 
---

## Abstract

`turf` is a Python package that fits Bayesian hierarchical Poisson or Negative Binomial regression models on the results of NFL or NHL games. These models infer home field advantage and the offensive and defensive strengths of the home and away teams from their past game results via Bayesian inference. Once the posterior distributions of model parameters has been inferred, samples drawn from the posterior can be used to simulate future games, estimate team strength of schedules, and quantify teams' offensive and defensive strengths, with uncertainties. Here, I detail the structure of the hierarcical models implemented in `turf`. All models are inspired by [Baio and Blangiardo (2010)](https://doi.org/10.1080/02664760802684177) in that they are log-linear Poisson (or Negative Binomial) regressions of home and away offensive and defensive performance fit on past results, i.e., home and away points. 

## Independent Poisson

The first model we consider is similar to the model considered in Section 2 of [Baio and Blangiardo (2010)](https://doi.org/10.1080/02664760802684177). We assume the home and away points in the $i^{th}$ game, $y_{h,i}$ and $y_{a,i}$, respectively, are modeled as conditionally-independent Poisson random variables. A Poisson distribution is suitable for game scores as it models the probability of observing a number of events, like points, in a given time interval, like a game. This model is given by,

$$
y_{h,i} | \theta_{h,i} \sim \mathrm{Poisson}(\theta_{h,i})
$$
where $\theta_{h,i}$ is the scoring intensity for the home team in the $i^{th}$ game. The same quantities are modeled in the same way for the away team.

For each team and game, the log scoring intensities are given by

$$
\log \theta_{h,i} = \mathrm{intercept} + \mathrm{home} + \mathrm{att}_{h,i} + \mathrm{def}_{a,i}
$$
$$
\log \theta_{a,i} = \mathrm{intercept} + \mathrm{att}_{a,i} + \mathrm{def}_{h,i}
$$

where the intercept term gives the typical amount of points scored by the average team and home indicates the additional points the home team typically gets due to not having to travel, fans, potentially favorable referee opinions, and more. $\mathrm{att}_{h,i}$ and $\mathrm{def}_{a,i}$ represent the attacking and defensive ability of the home and away teams in the $i^{th}$, respectively. This formalism effectively enables us to account for how a team's offense and their opponent's defense interact, for both home and away teams. 

We assume the following priors for the intercept and home advantage random parameters

$$
\mathrm{intercept} \sim \mathcal{N}(0,1)
$$
and
$$
\mathrm{home} \sim \mathcal{N}(0,1).
$$

Individual team effects are model as exchangeable random variables sampled from a common parent distribution (hyperprior). We use the following non-centered variables

$$
\mathrm{att}_x \sim \mu_{att,x} + \mathcal{N}(0,1) * \sigma_{att,x}
$$
$$
\mathrm{def}_x \sim \mu_{def,x} + \mathcal{N}(0,1) * \sigma_{def,x}
$$
for the $i^{th}$ team.

Good teams have high att (+ scoring intensity) and low def parameters (- scoring intensity imposed on opponent). Bad teams display the opposite.

We enforce a "sum-to-zero" constraint
$$
\sum_{x \in teams} \mathrm{att_x} = 0
$$
$$
\sum_{x \in teams} \mathrm{def_x} = 0
$$
for parameter identifiability and interpretability.

The hyperpriors on the group attacking and defensive strengths are modeled as
$$
\mu_{att} \sim \mathcal{N}(0, 1)
$$
$$
\mu_{def} \sim \mathcal{N}(0, 1)
$$
$$
\sigma_{att} \sim \mathrm{Gamma}(\alpha=2, \beta=0.1)
$$
$$
\sigma_{def} \sim \mathrm{Gamma}(\alpha=2, \beta=0.1).
$$




We use a hierarchical structure for this model by assuming that attacking and defensive strengths for each team are drawn from common parent distributions. We perform hierarchical Bayesian inference using `pymc` to infer posterior distributions for the parameters specified above. For more discussion on this type of model, see [Baio and Blangiardo (2010)](https://doi.org/10.1080/02664760802684177) and references therein.

- some
- bullet
- points

Footnotes can be entered using this code[^1].

[^1]: a footnote

![This is gonna be the caption.](pics/dummy.pdf){#fig:dummy width=40% height=20%}

## Discussion

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## References

[Baio and Blangiardo (2010)](https://doi.org/10.1080/02664760802684177)