"""

Test building models with turf

@author: David Fleming, 2024

"""


import os
from importlib import resources
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
    path = os.path.join(resources.files("turf.data"), "nfl_sample.csv")
    season = scrape.NFLSeason(year=2020, path=path)

    # Try building all the models
    model0 = inference.IndependentPoisson(season)
    model1 = inference.CorrelatedPoisson(season)
    model2 = inference.IndependentNegativeBinomial(season)
    model3 = inference.IndependentNegativeBinomialMixture(season)
    print(model0, model1, model2, model3)


if __name__ == "__main__":
    test_build_models()
