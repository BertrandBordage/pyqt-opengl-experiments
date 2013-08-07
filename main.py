#!/usr/bin/env python

import sys
from PyQt4 import QtGui
from engine import Window


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    win = Window()
    win.show()

    sys.exit(app.exec_())
