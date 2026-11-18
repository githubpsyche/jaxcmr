"""Each model module supplies a `make_factory` function. It binds the component
creators and returns a factory class. Construct that class with the dataset
and any vocabulary features or similarities. Its `create_trial_model`
method takes a trial index and a parameter mapping.

The semantic factories prepare a connection matrix for each trial. The
emotional-source factories prepare a category for each local item. Both use
the selected trial's item codes when constructing its model.
"""

__all__ = []
