# coding: utf-8

import numpy as np
from scipy.spatial import Voronoi
from utils import save_to_img


def distances_from_array(array, value):
    d = array - value
    return d[:, 0] ** 2 + d[:, 1] ** 2


def voronoi_matrix(size, n_points=20, coefs=(-1, 1), save=True):
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
        save_to_img(m)

    # We should return np.sqrt(m) but to optimize, we skip this step.
    return m
