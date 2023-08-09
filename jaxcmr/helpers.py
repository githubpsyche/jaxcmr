from copy import copy

def replace(instance, **kwargs):
    new_instance = copy(instance)

    for attr, value in kwargs.items():
        setattr(new_instance, attr, value)

    return new_instance