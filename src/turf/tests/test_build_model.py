#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test building models with turf

@author: David Fleming, 2023

"""


import os
from turf import scrape, inference


def test_build_models():
    """
    Test simply importing the module

    Parameters
    ----------
    None

    Returns
    -------
    """

    # Pull data locally
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, 'test_data', 'nfl_sample.csv')
    season = scrape.NFLSeason(year=2020, path=path)

    # Try building all the models
    model0 = inference.IndependentPoisson(season)
    model1 = inference.CorrelatedPoisson(season)
    model2 = inference.IndependentNegativeBinomial(season)
    model3 = inference.IndependentNegativeBinomialMixture(season)


if __name__ == "__main__":
    test_build_models()
