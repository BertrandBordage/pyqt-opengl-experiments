# coding: utf-8

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: c_string_type=bytes

from __future__ import unicode_literals, division
from libc.locale cimport setlocale, LC_NUMERIC
from libc.math cimport fabs, fmax, sqrt, cos, sin, M_PI, M_PI_2
from libc.stdio cimport puts, printf
import datetime

from numpy cimport (
    ndarray, import_array, PyArray_SimpleNewFromData, PyArray_Arange,
    PyArray_ZEROS, PyArray_Concatenate, PyArray_SwapAxes, PyArray_Reshape,
    NPY_FLOAT, NPY_UINT32)
from numpy import (array as np_array, sqrt as np_sqrt, rollaxis,
                   indices, column_stack, tile)
from numpy.random import randint as np_randint
from PIL import Image
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4 import QtGui
from PyQt4 import QtOpenGL
from PyQt4.QtGui import (
    QPixmap, QCursor, QSlider, QGroupBox, QGridLayout, QLabel, QDockWidget)
from scipy.ndimage import generic_filter

from diamond_square cimport continuous_map
from utils cimport equalize_height_map
from perturbation cimport perturbate_array
from voronoi cimport voronoi_array


import_array()


cdef extern from 'GL/gl.h' nogil:
    ctypedef unsigned int GLenum
    ctypedef unsigned int GLbitfield
    ctypedef void         GLvoid
    ctypedef int          GLint
    ctypedef unsigned int GLuint
    ctypedef int          GLsizei
    ctypedef float        GLfloat
    ctypedef double       GLdouble
    int GL_TRUE
    int GL_UNSIGNED_BYTE
    int GL_INT
    int GL_UNSIGNED_INT
    int GL_FLOAT
    int GL_COLOR_BUFFER_BIT
    int GL_DEPTH_BUFFER_BIT
    int GL_VERTEX_ARRAY
    int GL_NORMAL_ARRAY
    int GL_TEXTURE_COORD_ARRAY
    int GL_MODELVIEW
    int GL_PROJECTION
    int GL_DEPTH_TEST
    int GL_LIGHTING
    int GL_LIGHT_MODEL_LOCAL_VIEWER
    int GL_LIGHT0
    int GL_TEXTURE_2D
    int GL_TEXTURE_MAG_FILTER
    int GL_TEXTURE_MIN_FILTER
    int GL_RGB
    int GL_NEAREST
    int GL_COLOR_MATERIAL
    int GL_SPOT_CUTOFF
    int GL_QUADRATIC_ATTENUATION
    int GL_DIFFUSE
    int GL_SPECULAR
    int GL_SHININESS
    int GL_POSITION
    int GL_SPOT_DIRECTION
    int GL_AMBIENT_AND_DIFFUSE
    int GL_FRONT
    int GL_TRIANGLES
    void glClear(GLbitfield mask)
    void glEnable(GLenum cap)
    void glLightModeli(GLenum pname, GLint param)
    void glLightf(GLenum light, GLenum pname, GLfloat param)
    void glLightfv(GLenum light, GLenum pname, GLfloat *params)
    void glLighti(GLenum light, GLenum pname, GLint param)
    void glColorMaterial(GLenum face, GLenum mode)
    void glMaterialf(GLenum face, GLenum pname, GLfloat param)
    void glMaterialfv(GLenum face, GLenum pname, GLfloat *params)
    void glTexParameterf(GLenum target, GLenum pname, GLfloat param)
    void glTexImage2D(GLenum target, GLint level, GLint internalFormat,
                      GLsizei width, GLsizei height, GLint border,
                      GLenum format, GLenum type, GLvoid *pixels)
    void glMatrixMode(GLenum mode)
    void glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
    void glLoadIdentity()
    void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
    void glTranslatef(GLfloat x, GLfloat y, GLfloat z)
    void glEnableClientState(GLenum cap)
    void glVertexPointer(GLint size, GLenum type, GLsizei stride, GLvoid *ptr)
    void glNormalPointer(GLenum type, GLsizei stride, GLvoid *ptr)
    void glTexCoordPointer(GLint size, GLenum type, GLsizei stride,
                           GLvoid *ptr)
    void glDrawElements(GLenum mode, GLsizei count, GLenum type,
                        GLvoid *indices)

cdef extern from 'GL/glu.h' nogil:
    void gluPerspective(GLdouble fovy, GLdouble aspect,
                        GLdouble zNear, GLdouble zFar)
    void gluLookAt(GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ,
                   GLdouble centerX, GLdouble centerY, GLdouble centerZ,
                   GLdouble upX, GLdouble upY, GLdouble upZ)


cdef class TextureImage(object):
    cdef img
    cdef public:
        int width, height
        bytes str
        ndarray array
        char* array_ptr

    def __cinit__(self, filename):
        self.img = Image.open(filename)
        self.width, self.height = self.img.size
        self.str = self.img.tostring()
        cdef ndarray[char, ndim=1] array = np_array(list(self.str))
        self.array = array
        self.array_ptr = &array[0]


cdef inline float limit_float(float f, float m, float M) nogil:
    return m if f < m else M if f > M else f


cdef struct Coords:
    float x, y, z


cdef class Camera(object):
    cdef public float x, y, z, dx, dy, dz, adx, ady

    def __cinit__(self):
        self.x = 0.0
        self.y = 2.5
        self.z = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.adx = 0.0  # degrés
        self.ady = 0.0  # degrés

    cdef inline float arx(self) nogil:
        """
        X axis angle, in radians.
        """
        return self.adx * M_PI / 180.0

    cdef inline float ary(self) nogil:
        """
        Y axis angle, in radians.
        """
        return self.ady * M_PI / 180.0

    cdef inline Coords position(self) nogil:
        return [self.x, self.y, self.z]

    cdef inline Coords direction(self, float drx=0.0) nogil:
        cdef float arx = self.arx() + drx
        cdef float ary = self.ary()
        return [sin(ary) * cos(arx), sin(arx), cos(ary) * cos(arx)]

    cdef void update_gl(self) nogil:
        p = self.position()
        d = self.direction()
        up = self.direction(M_PI_2)
        gluLookAt(p.x, p.y, p.z,
                  p.x + d.x, p.y + d.y, p.z + d.z,
                  up.x, up.y, up.z)

    cdef void update(self) nogil:
        self.adx = limit_float(self.adx, -90.0, 90.0)
        self.ady %= 360.0
        cdef float a = self.ary()
        cdef float a_side = a + M_PI_2
        self.x += self.dz * sin(a) + self.dx * sin(a_side)
        self.y += self.dy
        self.z += self.dz * cos(a) + self.dx * cos(a_side)

    cdef unicode get_status(self):
        return 'x: %.2f  y: %.2f  z: %.2f  rotx: %.2f  roty: %.2f' % (
            self.x, self.y, self.z, self.adx, self.ady)



def get_slope(ndarray[double, ndim=1] a):
    return fmax(fmax(fabs(a[0] - a[2]),
                     fabs(a[1] - a[2])),
                fmax(fabs(a[3] - a[2]),
                     fabs(a[4] - a[2])))


cdef float erosion_score(ndarray[double, ndim=2] height_map, int size):
    cdef ndarray[unsigned int, ndim=2] footprint = np_array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], dtype=b'uint32')
    cdef ndarray[double, ndim=2] slope_map = generic_filter(
        height_map, get_slope, footprint=footprint)

    cdef float slope = slope_map.sum() / size ** 2
    cdef float standard_derivation = sqrt(
        ((slope_map - slope) ** 2).sum() / size ** 2)
    return standard_derivation / slope


cdef ndarray[double, ndim=2] build_height_map(size):
    cdef ndarray[double, ndim=2] height_map = equalize_height_map(
        continuous_map(size), -20.0, 20.0)
    height_map += equalize_height_map(voronoi_array(size), -17.0, 17.0)
    height_map = perturbate_array(height_map)
    # printf('Erosion score : %f\n', erosion_score(height_map, size))
    return height_map


cdef void normalize_vectors(ndarray[float, ndim=2] vectors):
    cdef ndarray[float, ndim=2] squared = vectors ** 2
    cdef ndarray[float, ndim=1] lens = np_sqrt(
        squared[:, 0] + squared[:, 1] + squared[:, 2])
    vectors[:, 0] /= lens
    vectors[:, 1] /= lens
    vectors[:, 2] /= lens


cdef ndarray[float, ndim=2] cross_product(
        ndarray[float, ndim=2] a, ndarray[float, ndim=2] b):
    a = PyArray_SwapAxes(a, 0, 1)
    b = PyArray_SwapAxes(b, 0, 1)
    cdef ndarray a0 = a[0], a1 = a[1], a2 = a[2], \
                 b0 = b[0], b1 = b[1], b2 = b[2]
    return PyArray_SwapAxes(
        PyArray_Reshape(
            PyArray_Concatenate(
                [a1*b2 - a2*b1,
                 a2*b0 - a0*b2,
                 a0*b1 - a1*b0], 0), [3, -1]), 0, 1)


cdef class Mesh(object):
    cdef int n
    cdef ndarray vertices, normals, texcoords, indices
    cdef float* vertices_ptr
    cdef float* normals_ptr
    cdef int* texcoords_ptr
    cdef unsigned int* indices_ptr
    cdef int indices_len

    def __cinit__(self, n):
        self.n = n
        setlocale(LC_NUMERIC, 'en_US.UTF-8')

        start = datetime.datetime.now()
        puts('Creating world…')

        self.create_vertices(n)
        vertices_time = datetime.datetime.now()
        printf('Vertices generated in %f seconds.\n',
               <double> (vertices_time - start).total_seconds())

        self.create_polygons(n)
        polygons_time = datetime.datetime.now()
        printf('Polygons loaded in %f seconds.\n',
               <double> (polygons_time - vertices_time).total_seconds())

        self.create_normals(n)
        normals_time = datetime.datetime.now()
        printf('Normal vectors calculated in %f seconds.\n',
               <double> (normals_time - polygons_time).total_seconds())

        self.create_texture_coordinates(n)
        texture_time = datetime.datetime.now()
        printf('Textures loaded in %f seconds.\n',
               <double> (texture_time - normals_time).total_seconds())

        printf('Total loading in %f seconds.\n',
               <double> (texture_time - start).total_seconds())

    cdef void create_vertices(self, int n):
        cdef ndarray[float, ndim=2] indices_xz
        # Taken from http://stackoverflow.com/a/4714857/1576438
        indices_xz = PyArray_Arange(-n // 2, n // 2, 1, NPY_FLOAT)[
            PyArray_Reshape(rollaxis(indices([n, n]), 0, 3), [-1, 2])]
        cdef ndarray[float, ndim=2] vertices = column_stack((
            indices_xz[:, 0],
            build_height_map(n).astype(b'float32').flatten(),
            indices_xz[:, 1]))
        self.vertices = vertices

        # Builds a pointer for optimization.
        self.vertices_ptr = &vertices[0, 0]

    cdef void create_polygons(self, int n):
        cdef unsigned int* two_triangles = [0, 1, n, 1, n+1, n]
        cdef ndarray[unsigned int, ndim=2] indices = (
            (PyArray_SimpleNewFromData(1, [6], NPY_UINT32, two_triangles)
             + PyArray_Arange(0, n - 1, 1, NPY_UINT32).reshape(-1, 1)).flatten()
            + PyArray_Arange(0, (n - 1) ** 2, n, NPY_UINT32).reshape(-1, 1)
        ).reshape(-1, 3)

        # Builds a pointer for optimization.
        self.indices = indices
        self.indices_ptr = &indices[0, 0]

        self.indices_len = indices.shape[0] * indices.shape[1]

    cdef void create_normals(self, int n):
        # Taken from https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
        cdef ndarray[float, ndim=2] normals = PyArray_ZEROS(
            2, [n ** 2, 3], NPY_FLOAT, 0)
        indices = self.indices
        cdef ndarray[float, ndim=3] faces = PyArray_SwapAxes(
            self.vertices[indices], 0, 1)
        first_vertices = faces[0]
        cdef ndarray[float, ndim=2] normals_per_face = cross_product(
            faces[1] - first_vertices, faces[2] - first_vertices)
        indices = PyArray_SwapAxes(indices, 0, 1)
        normals[indices[0]] += normals_per_face
        normals[indices[1]] += normals_per_face
        normals[indices[2]] += normals_per_face
        normalize_vectors(normals)

        # Builds a pointer for optimization.
        self.normals = normals
        self.normals_ptr = &normals[0, 0]

    cdef void create_texture_coordinates(self, int n):
        cdef ndarray[int, ndim=3] texcoords = tile(
            PyArray_Concatenate([
                tile([[0, 0], [0, 1]], (n // 2, 1, 1)),
                tile([[1, 0], [1, 1]], (n // 2, 1, 1))], 0),
            (n // 2, 1, 1)).astype(b'int32')

        # Builds a pointer for optimization.
        self.texcoords = texcoords
        self.texcoords_ptr = &texcoords[0, 0, 0]


cdef class World(object):
    cdef public parent
    cdef int n
    cdef TextureImage texture
    cdef Mesh mesh
    
    cdef public Camera camera
    cdef Coords spot_position, spot_direction
    cdef public bint fixed_spot, new_fixed_spot
    cdef public action
    cdef public float action_step

    def __cinit__(self, parent):
        self.parent = parent
        self.texture = TextureImage('texture.png')

        self.camera = Camera()
        self.fixed_spot = False
        self.new_fixed_spot = True

        self.action = None
        self.action_step = 1.0

        self.n = n = 512
        self.mesh = Mesh(n)

    def initialize_gl(self):
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 0.5])
        glMaterialf(GL_FRONT, GL_SHININESS, 100.0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.00001)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 0.7])

        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D,
                     0, GL_RGB, self.texture.width, self.texture.height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, self.texture.array_ptr)

    def update_gl(self):
        self.camera.update()

        cdef int spot_cutoff = self.parent.parent.spot_slider.value()
        cdef float aspect = (self.parent.parent.width
                             / self.parent.parent.height), \
                   fov = self.parent.parent.fov_slider.value()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        spot_position = self.camera.position()
        if self.fixed_spot:
            if self.new_fixed_spot:
                self.spot_position = spot_position
            spot_position = self.spot_position
        p = spot_position
        glLightfv(GL_LIGHT0, GL_POSITION, [p.x, p.y, p.z, 1.0])

        spot_direction = self.camera.direction()
        if self.fixed_spot:
            if self.new_fixed_spot:
                self.spot_direction = spot_direction
                self.new_fixed_spot = False
            spot_direction = self.spot_direction
        d = spot_direction
        glLighti(GL_LIGHT0, GL_SPOT_CUTOFF, spot_cutoff)
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, [d.x, d.y, d.z])

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(fov, aspect, 1.0, 100000.0)
        self.camera.update_gl()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.mesh.vertices_ptr)

        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, self.mesh.normals_ptr)

        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_INT, 0, self.mesh.texcoords_ptr)

        glDrawElements(GL_TRIANGLES, self.mesh.indices_len,
                       GL_UNSIGNED_INT, self.mesh.indices_ptr)

    def update(self):
        cdef ndarray[float, ndim=2] vertices = self.mesh.vertices
        cdef ndarray[long, ndim=1] random_vertices
        cdef int n
        cdef long i
        DEF moved_vertices = 20000

        if self.action is not None:
            random_vertices = np_randint(
                len(vertices), size=moved_vertices)
            vertices[random_vertices] += [0.0, self.action, 0.0]

        # Make the y coordinate of the camera follow the mesh.
        ys = vertices[:, 1].reshape(-1, self.n)
        hn = self.n // 2
        try:
            result = 10 + ys[hn + self.camera.x, hn + self.camera.z]
        except IndexError:
            pass
        else:
            diff = self.camera.y - result
            self.camera.y -= diff / 4 if diff < 0 else diff / 20

    cdef int get_polygon_count(self):
        return self.mesh.indices_len / 3  # 3 points par face.

    def get_status(self):
        return '%s polygons: %d' % (self.camera.get_status(),
                                    self.get_polygon_count())


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        self.world = World(self)
        self.frames_counted = 0
        self.fps_iterations = 0
        super(GLWidget, self).__init__(parent)

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(0, 0, 150))

        self.world.initialize_gl()

        self.last_time = datetime.datetime.now()
        self.current_time = datetime.datetime.now()

    def resizeGL(self, width, height):
        if height == 0:
            height = 1

        glViewport(0, 0, width, height)
        self.parent.width = width
        self.parent.height = height

    def paintGL(self):
        self.world.update_gl()

    def updateDispatcher(self):
        self.updateGL()
        self.fps_iterations += 1

    def updateStatusBar(self):
        self.updateFPS()
        seconds_elapsed = (self.current_time - self.last_time).total_seconds()
        self.parent.statusBar().showMessage(
            'FPS: %.2f %s' % (self.frames_counted / seconds_elapsed,
                              self.world.get_status()))

    def updateFPS(self):
        self.last_time = self.current_time
        self.current_time = datetime.datetime.now()
        self.frames_counted = self.fps_iterations
        self.fps_iterations = 0


class Window(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.width = 1366
        self.height = 768
        self.resize(self.width, self.height)
        self.setWindowTitle('OpenGL experiments')

        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)

        self.initActions()
        self.initMenus()

        self.mouse_x = None
        self.mouse_y = None
        self.mouse_locked = False

        display_timer = QtCore.QTimer(self)
        QtCore.QObject.connect(display_timer, QtCore.SIGNAL('timeout()'),
                               self.glWidget.updateDispatcher)
        display_timer.start(1000 / 60)

        world_timer = QtCore.QTimer(self)
        QtCore.QObject.connect(world_timer, QtCore.SIGNAL('timeout()'),
                               self.glWidget.world.update)
        world_timer.start(1000 / 60)

        statusbar_timer = QtCore.QTimer(self)
        QtCore.QObject.connect(statusbar_timer, QtCore.SIGNAL('timeout()'),
                               self.glWidget.updateStatusBar)
        statusbar_timer.start(1000)

    def lockMouse(self):
        self.mouse_locked = True
        self.setMouseTracking(True)
        self.setFocus()

        px = QPixmap(32, 32)
        px.fill()
        px.setMask(px.createHeuristicMask())

        self.setCursor(QCursor(px))
        self.grabMouse(self.cursor())

    def unlockMouse(self):
        self.releaseMouse()
        self.setMouseTracking(False)
        self.setCursor(Qt.ArrowCursor)

        self.mouse_x = None
        self.mouse_y = None
        self.mouse_locked = False

    def initActions(self):
        self.fullscreenAction = QtGui.QAction('Full&screen', self)
        self.fullscreenAction.setShortcut('F11')
        self.exitAction = QtGui.QAction('&Quit', self)

        def full():
            self.unlockMouse()
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            self.lockMouse()

        self.connect(self.fullscreenAction, QtCore.SIGNAL('triggered()'), full)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), self.close)

    def initMenus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(self.fullscreenAction)
        file_menu.addAction(self.exitAction)

        self.controls = QGroupBox()
        controls_layout = QGridLayout()

        self.text = QtGui.QTextEdit("""
        <b>ZQSD</b>: Move horizontally<br/>
        <b>Space</b>: Move up<br/>
        <b>Shift</b>: Move down<br/><br/>

        <b>Left click</b>: Move thousands of vertices<br/>
        <b>Right click</b>: Move them in the opposite direction<br/>
        <b>Mouse wheel</b>: Change vertices movement speed<br/><br/>

        <b>Mouse wheel click</b>: Freeze|Unfreeze light source<br/><br/>

        <b>Escape</b>: Ungrab mouse
        """)
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(350)
        controls_layout.addWidget(self.text, 0, 0, 1, 2)

        controls_layout.addWidget(QLabel('Field\nof view'), 1, 0)
        self.fov_slider = QSlider(Qt.Horizontal)
        self.fov_slider.setRange(30, 90)
        self.fov_slider.setValue(45)
        controls_layout.addWidget(self.fov_slider, 1, 1)

        controls_layout.addWidget(QLabel('Camera\nspeed'), 2, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(2)
        controls_layout.addWidget(self.speed_slider, 2, 1)

        controls_layout.addWidget(QLabel('Spot\ncutoff'), 3, 0)
        self.spot_slider = QSlider(Qt.Horizontal)
        self.spot_slider.setRange(1, 90)
        self.spot_slider.setValue(15)
        controls_layout.addWidget(self.spot_slider, 3, 1)

        self.controls.setLayout(controls_layout)
        dock = QDockWidget('Controls')
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        dock.setWidget(self.controls)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.isAutoRepeat():
            return
        key = QKeyEvent.key()
        move_amount = self.speed_slider.value()
        cam = self.glWidget.world.camera
        if key == Qt.Key_Z:
            cam.dz += move_amount
        if key == Qt.Key_S:
            cam.dz -= move_amount
        if key == Qt.Key_Q:
            cam.dx += move_amount
        if key == Qt.Key_D:
            cam.dx -= move_amount
        if key == Qt.Key_Space:
            cam.dy += move_amount
        if key == Qt.Key_Shift:
            cam.dy -= move_amount
        if key == Qt.Key_Escape:
            self.unlockMouse()

    def keyReleaseEvent(self, QKeyEvent):
        if QKeyEvent.isAutoRepeat():
            return
        key = QKeyEvent.key()
        move_amount = self.speed_slider.value()
        cam = self.glWidget.world.camera
        if key == Qt.Key_Z:
            cam.dz -= move_amount
        if key == Qt.Key_S:
            cam.dz += move_amount
        if key == Qt.Key_Q:
            cam.dx -= move_amount
        if key == Qt.Key_D:
            cam.dx += move_amount
        if key == Qt.Key_Space:
            cam.dy -= move_amount
        if key == Qt.Key_Shift:
            cam.dy += move_amount

    def mouseMoveEvent(self, QMouseEvent):
        mouse_x = QMouseEvent.x()
        mouse_y = QMouseEvent.y()

        if self.mouse_x is None and self.mouse_y is None:
            p = self.mapToGlobal(QMouseEvent.pos())
            self.global_mouse_x, self.global_mouse_y = p.x(), p.y()
            self.mouse_x = mouse_x
            self.mouse_y = mouse_y

        if self.mouse_locked:
            self.cursor().setPos(self.global_mouse_x, self.global_mouse_y)

        self.glWidget.world.camera.ady -= (mouse_x - self.mouse_x) / 10.0
        self.glWidget.world.camera.adx -= (mouse_y - self.mouse_y) / 10.0

    def wheelEvent(self, QWheelEvent):
        self.glWidget.world.action_step += QWheelEvent.delta() / 120.0
        self.mousePressEvent(QWheelEvent)

    def mousePressEvent(self, QMouseEvent):
        if not self.mouse_locked:
            self.lockMouse()
            return
        button = QMouseEvent.buttons()
        action = 0
        action_step = self.glWidget.world.action_step
        if button == Qt.LeftButton:
            action += action_step
        if button == Qt.RightButton:
            action -= action_step
        if button == Qt.MiddleButton:
            self.glWidget.world.fixed_spot = not self.glWidget.world.fixed_spot
            if self.glWidget.world.fixed_spot:
                self.glWidget.world.new_fixed_spot = True
        self.glWidget.world.action = action or None

    def mouseReleaseEvent(self, QMouseEvent):
        self.glWidget.world.action = None

    def close(self):
        QtGui.qApp.quit()
