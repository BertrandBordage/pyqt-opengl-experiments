from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension('utils', ['utils.pyx']),
    Extension('diamond_square', ['diamond_square.pyx']),
    Extension('voronoi', ['voronoi.pyx']),
    Extension('perturbation', ['perturbation.pyx']),
    Extension('engine', ['engine.pyx'], libraries=['GL', 'GLU']),
]


setup(
    name='PyQt OpenGL experiments',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
