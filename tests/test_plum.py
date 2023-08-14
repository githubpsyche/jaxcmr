from plum import dispatch, NotFoundLookupError
import pytest

# %% Partial Type Annotation

@dispatch
def add(x: int, y: int) -> int:
    return x + y

@dispatch
def add(x: int, y):
    return x + y + y

def test_partial_type_annotation():
    assert add(1, 2) == 3
    assert add(1, 2.0) == 5.0
    with pytest.raises(NotFoundLookupError):
        add(1.0, 2.0)


# %% Variable Defaults

@dispatch
def add_and_power(x: int, y: int = 0, z: int = 1):
    return (x + y) ** z

def test_variable_defaults():
    assert add_and_power(1) == 1
    assert add_and_power(1, 2) == 3
    assert add_and_power(1, 2, 3) == 27
    assert add_and_power(1, z=3) == 1
    assert add_and_power(1, z=3, y=2) == 27