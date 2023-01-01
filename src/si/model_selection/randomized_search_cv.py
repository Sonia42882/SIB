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
    scores:
        The scores of the model on the dataset.
    """

    scores = {
        'parameters': [],
        'seed': [],
        'train': [],
        'test': []
    }

    # verificar se parametros fornecidos existem no modelo
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    #  Obtém n_iter combinações de parâmetros
    for i in range(n_iter):

        random_state = np.random.randint(0, 1000)
        scores['seed'].append(random_state)
        parameters = {}
        for parameter, value in parameter_distribution.items():
            parameters[parameter] = np.random.choice(value)

        for parameter, value in parameters.items():
            setattr(model, parameter, value)

    # cross_validation
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        scores['parameters'].append(parameters)
        scores['train'].append(score['train'])
        scores['test'].append(score['test'])

    return scores
