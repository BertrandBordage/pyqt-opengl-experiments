# coding: utf-8

from __future__ import unicode_literals, division
from itertools import product
from random import uniform
import numpy as np
from PIL import Image
import datetime


i = 0


def save_to_img(m):
    global i
    mini = m.min()
    if mini < 0:
        m += - mini
    m *= 255 / m.max()

    img = Image.fromarray(m.astype(b'uint8'))
    img.save('diamond_square/truc%s.png' % i)
    i += 1


def _get_real_min(mini, size):
    if -size < mini < 0:
        return mini - 1
    return mini


def _get_real_max(maxi, size):
    if maxi >= size:
        return maxi - (size - 1)
    return maxi


def get_inf_sup_coords(x, y, step, size):
    xmin = _get_real_min(x - step, size)
    ymin = _get_real_min(y - step, size)
    xmax = _get_real_max(x + step, size)
    ymax = _get_real_max(y + step, size)
    return xmin, xmax, ymin, ymax


def build_height_map(size, amplitude=10, smoothing=10, save=True):
    orig_size = size
    size += (size + 1) % 2
    m = np.zeros((size, size))

    random_func = lambda: uniform(-1, 1)

    step = size // 2
    while step:
        random_coef = step / smoothing
        # Diamond
        for x, y in product(*(range(step, size, step*2),) * 2):
            xmin, xmax, ymin, ymax = get_inf_sup_coords(x, y, step, size)
            m[x, y] = (m[xmin, ymin] + m[xmax, ymin]
                       + m[xmin, ymax] + m[xmax, ymax]
                       + random_func() * random_coef) / 4
        # Square
        for x, y in product(*(range(0, size, step),) * 2):
            if m[x, y]:
                continue
            xmin, xmax, ymin, ymax = get_inf_sup_coords(x, y, step, size)
            m[x, y] = (m[x, ymin] + m[xmin, y]
                       + m[xmax, y] + m[x, ymax]) / 4
        if step == 1:
            break
        step //= 2

    m = m[:orig_size, :orig_size] * amplitude

    if save:
        save_to_img(m.copy())

    return m

if __name__ == '__main__':
    for i in range(50):
        start = datetime.datetime.now()
        build_height_map(256)
        print((datetime.datetime.now() - start).total_seconds())
