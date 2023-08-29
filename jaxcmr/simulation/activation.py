import matplotlib.pyplot as plt
import numpy as np

from jaxcmr.helpers import Float, Array


def visualize_2d_array(array: Float[Array, "input_features output_features"]):
    array = np.array(array)
    rows, cols = array.shape

    plt.figure(figsize=(cols, rows))  # set figure size to avoid overlap in strings
    plt.imshow(array, cmap="viridis")
    plt.colorbar()

    # Add value annotations on each cell of the array
    for i in range(rows):
        for j in range(cols):
            plt.text(
                j, i, format(array[i, j], ".3f"), ha="center", va="center", color="w"
            )

    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(rows))
    plt.xlabel("output")
    plt.ylabel("input")
    plt.title("2D Array Visualization")
    plt.show()
