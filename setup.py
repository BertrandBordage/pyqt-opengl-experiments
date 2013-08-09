from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules = [
    Extension('diamond_square', ['diamond_square.pyx']),
    Extension('engine', ['engine.pyx'], libraries=['GL']),
]


setup(
    name='PyQt OpenGL experiments',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
