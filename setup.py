from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


def ext(name, numpy=False, GL=False, GLU=False, OpenMP=False):
    kwargs = {
        'name': name,
        'sources': [name + '.pyx'],
        'include_dirs': [],
        'libraries': [],
        'extra_compile_args': [],
        'extra_link_args': [],
    }
    if numpy:
        kwargs['include_dirs'].append(np.get_include())
    if GL:
        kwargs['libraries'].append('GL')
    if GLU:
        kwargs['libraries'].append('GLU')
    if OpenMP:
        kwargs['extra_compile_args'].append('-fopenmp')
        kwargs['extra_link_args'].append('-fopenmp')
    return Extension(**kwargs)


ext_modules = [
    ext('utils', numpy=True),
    ext('diamond_square', numpy=True),
    ext('voronoi', numpy=True),
    ext('perturbation', numpy=True),
    ext('engine', GL=True, GLU=True, numpy=True),
]


setup(
    name='PyQt OpenGL experiments',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
