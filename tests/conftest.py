import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_test_figures():
    yield
    plt.close("all")
