import itertools
import numpy as np
from typing import Callable, Tuple, Dict, List, Any

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate

def randomized_search_cv(model, dataset: Dataset, parameter_distribution: Dict[str, Tuple], scoring: Callable = None, cv: int = 5, n_iter: int = 10, test_size: float = 0.2):
    """
    Performs a grid search cross validation on a model.
    Parameters
    ----------
    model
        The model to cross validate
    dataset: Dataset
        The dataset to cross validate on
    parameter_distribution: Dict[str, Tuple]
        The parameters to use
    scoring: Callable
        The scoring function to use
    cv: int
        The cross validation folds
    n_iter: int
        The number of parameter random combinations
    test_size: float
        The test size
    Returns
    -------
    scores: List[Dict[str, List[float]]]
        The scores of the model on the dataset.
    """

    # verificar se parametros fornecidos existem no modelo
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []
    combinations = []

    # obtém n_iter combinações de parâmetros
    for i in range(n_iter):
        parameter_grid = {}
        for param in parameter_distribution.keys():
            parameter_grid[param] = np.random.choice(parameter_distribution[param])
        combinations.append(parameter_grid)

    # for each combination
    for combination in combinations:

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_distribution.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # add the parameter configuration
        score['parameters'] = parameters

        # add the score
        scores.append(score)

    return scores
