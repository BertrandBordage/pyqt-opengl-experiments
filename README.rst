PyQt OpenGL experiments
=======================

Ubuntu 13.04
------------

Install dependencies
....................

#. ``sudo apt-get install python-dev python-pip python-qt4 python-qt4-gl freeglut3-dev gfortran liblapack-dev``
#. ``sudo apt-get build-dep python-imaging``
#. ``sudo ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/``
#. ``sudo ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/``
#. ``sudo ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/``
#. ``sudo pip install -r requirements.txt``


Compile
.......

``python setup.py build_ext --inplace --force``


Run, you clever boy(girl)!
..........................

``./main.py``

And remember: `compile`_ after each update.
