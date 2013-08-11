from matplotlib import pyplot
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from utils import save_to_img


def distance(d):
    return np.sqrt(d[0] ** 2 + d[1] ** 2)


def find_nearest(array, value):
    d = np.abs(array - value)
    found = tuple(min(d, key=lambda a: distance(a)))
    for i, a in enumerate(d):
        if tuple(a) == found:
            return array[i]


def voronoi_matrix(size, n_points=30, save=False, plot=False):
    points = np.random.randint(0, size, (n_points, 2))

    v = Voronoi(points)

    m = np.zeros((size, size))
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            nearest = find_nearest(v.vertices, (x, y))
            d = nearest - (x, y)
            m[x, y] = -distance(d)

    if save:
        save_to_img(m.copy())
    if plot:
        voronoi_plot_2d(v)
        pyplot.show()

    return m
