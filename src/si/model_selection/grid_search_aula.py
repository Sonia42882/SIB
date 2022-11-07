

def grid_search_cv(model, dataset, parameter_grid, scoring, cv, test_size):
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
    scores = []
    for combination in itertools.product(* parameter_grid.values()): # * faz unpack
        parameters = {}
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)
#
#
# return

#ideia perceber cross validation e o que a procura por parametros faz
#
