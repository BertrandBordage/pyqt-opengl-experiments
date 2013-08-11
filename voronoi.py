from matplotlib import pyplot
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from utils import save_to_img


def distances_from_array(array, value):
    d = array - value
    return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)


def voronoi_matrix(size, n_points=15, coefs=(-1, 1), save=True, plot=False):
    points = np.random.randint(0, size, (n_points, 2))
    voronoi = Voronoi(points)

    m = np.zeros((size, size))
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            distances = distances_from_array(voronoi.vertices, (x, y))
            distances.sort()
            m[x, y] = sum(coef * distances[i]
                          for i, coef in enumerate(coefs) if coef)

    if save:
        save_to_img(m.copy())
    if plot:
        voronoi_plot_2d(voronoi)
        pyplot.show()

    return m
