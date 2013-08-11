from matplotlib import pyplot
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from utils import save_to_img


def distance_to_array(array, value):
    d = np.abs(array - value)
    distances = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    return distances.min()


def voronoi_matrix(size, n_points=30, save=True, plot=False):
    points = np.random.randint(0, size, (n_points, 2))

    v = Voronoi(points)

    m = np.zeros((size, size))
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            m[x, y] = -distance_to_array(v.vertices, (x, y))

    if save:
        save_to_img(m.copy())
    if plot:
        voronoi_plot_2d(v)
        pyplot.show()

    return m
