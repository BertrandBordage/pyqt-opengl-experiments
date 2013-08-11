# coding: utf-8

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: c_string_type=bytes

from __future__ import unicode_literals, division
from libc.math cimport cos, sin, M_PI, M_PI_2
from libc.stdio cimport puts, printf
import datetime

import numpy as np
cimport numpy as np
from PIL import Image
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PyQt4 import QtGui
from PyQt4 import QtOpenGL
from PyQt4.QtGui import (
    QPixmap, QCursor, QSlider, QGroupBox, QGridLayout, QLabel, QDockWidget)
from diamond_square cimport continuous_map
from utils cimport equalize_height_map, save_to_img
from perturbation cimport perturbate_array
from voronoi cimport voronoi_array


cdef extern from 'GL/gl.h' nogil:
    ctypedef unsigned int GLenum
    ctypedef unsigned int GLbitfield
    ctypedef void         GLvoid
    ctypedef int          GLint
    ctypedef unsigned int GLuint
    ctypedef int          GLsizei
    ctypedef float        GLfloat
    ctypedef double       GLdouble
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
    int GL_FRONT_AND_BACK
    int GL_QUADS
    void glClear(GLbitfield mask)
    void glEnable(GLenum cap)
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

cdef class TextureImage(object):
    cdef img
    cdef public:
        int width, height
        bytes str
        np.ndarray array
        char* array_ptr

    def __cinit__(self, filename):
        self.img = Image.open(filename)
        self.width, self.height = self.img.size
        self.str = self.img.tostring()
        cpdef np.ndarray[char, ndim=1] array = np.array(list(self.str))
        self.array = array
        self.array_ptr = &array[0]


cdef class Cube(object):
    cdef np.ndarray vertices, normals, texcoords, indices

    def __cinit__(self):
        self.vertices = np.array([
            1, 1, 0,  # Front  top    left
            0, 1, 0,  # Front  top    right
            0, 0, 0,  # Front  bottom right
            1, 0, 0,  # Front  bottom left

            0, 1, 1,  # Back   top    right
            1, 1, 1,  # Back   top    left
            1, 0, 1,  # Back   bottom left
            0, 0, 1,  # Back   bottom right

            1, 1, 1,  # Top    back   left
            0, 1, 1,  # Top    back   right
            0, 1, 0,  # Top    front  right
            1, 1, 0,  # Top    front  left

            1, 0, 0,  # Bottom front  left
            0, 0, 0,  # Bottom front  right
            0, 0, 1,  # Bottom back   right
            1, 0, 1,  # Bottom back   left

            1, 1, 1,  # Left   top    back
            1, 1, 0,  # Left   top    front
            1, 0, 0,  # Left   bottom front
            1, 0, 1,  # Left   bottom back

            0, 1, 0,  # Right  top    front
            0, 1, 1,  # Right  top    back
            0, 0, 1,  # Right  bottom back
            0, 0, 0,  # Right  bottom front
        ], dtype=b'float32').reshape(-1, 3)

        self.normals = np.array([
            # Front face
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            # Back face
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
            # Top face
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
            # Bottom face
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
            # Left face
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
            # Right face
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        ], dtype=b'float32')

        self.texcoords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]],
                                  dtype=b'int32')

        self.indices = np.array([
            0, 1, 2, 3,  # Front  face
            4, 5, 6, 7,  # Back   face
            8, 9, 10, 11,  # Top    face
            12, 13, 14, 15,  # Bottom face
            16, 17, 18, 19,  # Left   face
            20, 21, 22, 23,  # Right  face
        ], dtype=b'uint32')


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

    cdef Coords position(self) nogil:
        return [self.x, -self.y, self.z]

    cdef inline float arx(self) nogil:
        """
        Angle de l'axe x, en radians.
        """
        return self.adx * M_PI / 180.0

    cdef inline float ary(self) nogil:
        """
        Angle de l'axe y, en radians.
        """
        return self.ady * M_PI / 180.0

    cdef inline Coords get_spot_position(self) nogil:
        return [-self.x, self.y, -self.z]

    cdef inline Coords get_spot_direction(self) nogil:
        cdef float arx = self.arx()
        cdef float ary = self.ary()
        return [-sin(ary) * cos(arx), sin(arx), -cos(ary) * cos(arx)]

    cdef void update_gl(self) nogil:
        glRotatef(self.adx, -1.0, 0.0, 0.0)
        glRotatef(self.ady, 0.0, -1.0, 0.0)
        p = self.position()
        glTranslatef(p.x, p.y, p.z)

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


cdef class World(object):
    cdef public parent
    cdef Cube cube
    cdef TextureImage texture
    cdef readonly int per_cube
    cdef np.ndarray vertices, normals, texcoords, indices
    cdef float* vertices_ptr
    cdef float* normals_ptr
    cdef int* texcoords_ptr
    cdef unsigned int* indices_ptr
    cdef int indices_len
    
    cdef public Camera camera
    cdef Coords spot_position, spot_direction
    cdef bint fixed_spot, new_fixed_spot
    cdef public action
    cdef public float action_step

    def __cinit__(self, parent):
        self.parent = parent
        self.cube = Cube()
        self.texture = TextureImage('texture.png')

        self.camera = Camera()
        self.fixed_spot = False
        self.new_fixed_spot = True

        self.action = None
        self.action_step = 1.0        

    cdef void create_vertices(self, int n):
        cdef np.ndarray[float, ndim=2] cube_vertices, indices_xz, indices_xyz
        cdef np.ndarray[double, ndim=2] height_map
        cube_vertices = self.cube.vertices
        self.per_cube = len(cube_vertices)
        # Taken from http://stackoverflow.com/a/4714857/1576438
        indices_xz = np.arange(-n // 2, n // 2, dtype=b'float32')[
            np.rollaxis(np.indices((n,) * 2), 0, 2 + 1).reshape(-1, 2)]
        indices_xyz = np.zeros((n ** 2, 3), dtype=b'float32')
        indices_xyz[:, 0] = indices_xz[:, 0]

        height_map = equalize_height_map(continuous_map(n), -20.0, 20.0)
        height_map += equalize_height_map(voronoi_array(n), -17.0, 17.0)
        save_to_img(height_map)
        height_map = perturbate_array(height_map)
        indices_xyz[:, 1] = height_map.flatten()

        indices_xyz[:, 2] = indices_xz[:, 1]
        cdef np.ndarray[float, ndim=2] vertices = (
            cube_vertices + indices_xyz.reshape(-1, 1, 3)
        ).reshape(-1, 3)
        self.vertices = vertices

        # Builds a pointer for optimization.
        self.vertices_ptr = &vertices[0, 0]

    cdef void create_normals(self):
        cdef np.ndarray[float, ndim=1] normals = np.tile(
            self.cube.normals,
            (len(self.vertices) / self.per_cube))

        # Builds a pointer for optimization.
        self.normals = normals
        self.normals_ptr = &normals[0]

    cdef void create_texture_coordinates(self):
        cdef np.ndarray[int, ndim=3] texcoords = np.tile(
            self.cube.texcoords,
            (len(self.vertices) / self.per_cube, 6, 1))

        # Builds a pointer for optimization.
        self.texcoords = texcoords
        self.texcoords_ptr = &texcoords[0, 0, 0]

    cdef void create_polygons(self, int n):
        cdef np.ndarray[unsigned int, ndim=1] indices = (
             self.cube.indices + np.arange(
                 self.per_cube * n ** 2, step=self.per_cube,
                 dtype=b'uint32').reshape(-1, 1)
        ).flatten()

        # Builds a pointer for optimization.
        self.indices = indices
        self.indices_ptr = &indices[0]

        self.indices_len = len(self.indices)

    cdef void create(self):
        start = datetime.datetime.now()
        puts('Création du monde…')

        cdef int n = 256

        self.create_vertices(n)
        vertices_time = datetime.datetime.now()
        printf('Chargement des points terminé en %f secondes.\n',
               <double>(vertices_time - start).total_seconds())

        self.create_normals()
        normals_time = datetime.datetime.now()
        printf('Chargement des vecteurs normaux terminé en %f secondes.\n',
               <double>(normals_time - vertices_time).total_seconds())

        self.create_texture_coordinates()
        texture_time = datetime.datetime.now()
        printf('Chargement des textures terminé en %f secondes.\n',
               <double>(texture_time - normals_time).total_seconds())

        self.create_polygons(n)
        polygon_time = datetime.datetime.now()
        printf('Chargement des polygones terminé en %f secondes.\n',
               <double>(polygon_time - vertices_time).total_seconds())

        printf('Temps total de chargement : %f secondes.\n',
               <double>(polygon_time - start).total_seconds())

    def initialize_gl(self):
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 0.5])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.00005)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 0.7])

        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D,
                     0, GL_RGB, self.texture.width, self.texture.height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, self.texture.array_ptr)

        self.create()

    def update_gl(self):
        self.camera.update()

        cdef int spot_cutoff = self.parent.parent.spot_slider.value()
        cdef float aspect = (self.parent.parent.width
                             / self.parent.parent.height), \
                   fov = self.parent.parent.fov_slider.value()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        spot_position = self.camera.get_spot_position()
        if self.fixed_spot:
            if self.new_fixed_spot:
                self.spot_position = spot_position
            spot_position = self.spot_position
        p = spot_position
        glTranslatef(p.x, p.y, p.z)

        spot_direction = self.camera.get_spot_direction()
        if self.fixed_spot:
            if self.new_fixed_spot:
                self.spot_direction = spot_direction
                self.new_fixed_spot = False
            spot_direction = self.spot_direction
        d = spot_direction
        glLighti(GL_LIGHT0, GL_SPOT_CUTOFF, spot_cutoff)
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, [d.x, d.y, d.z])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(fov, aspect, 1.0, 100000.0)
        self.camera.update_gl()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices_ptr)

        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, self.normals_ptr)

        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_INT, 0, self.texcoords_ptr)

        glDrawElements(GL_QUADS, self.indices_len,
                       GL_UNSIGNED_INT, self.indices_ptr)

    def update(self):
        cdef tuple offset
        cdef np.ndarray[float, ndim=2] vertices = self.vertices
        cdef np.ndarray[long, ndim=1] random_cubes
        cdef int n
        cdef long i, per_cube = self.per_cube
        DEF moved_cubes = 500

        if self.action is not None:
            offset = (0.0, self.action, 0.0)
            random_cubes = np.random.randint(
                len(vertices) / per_cube, size=moved_cubes) * per_cube
            for n in range(moved_cubes):
                i = random_cubes[n]
                vertices[i:i + per_cube] += offset

    cdef int get_polygon_count(self):
        return len(self.indices) / 4  # 4 points par face.

    def get_status(self):
        return '%s polygones: %d' % (self.camera.get_status(),
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
            'fps: %.2f %s' % (self.frames_counted / seconds_elapsed,
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
        self.setWindowTitle('Expérience OpenGL')

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
        <b>ZQSD</b> : Déplacement<br/>
        <b>Espace</b> : Monter<br/>
        <b>Maj</b> : Descendre<br/><br/>

        <b>Clic gauche</b> : Déplacer des cubes par centaines<br/>
        <b>Clic droit</b> : Les déplacer dans l’autre sens<br/>
        <b>Molette</b> : Changer la vitesse de déplacement des cubes<br/><br/>

        <b>Échap</b> : Relâcher la souris
        """)
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(280)
        controls_layout.addWidget(self.text, 0, 0)

        controls_layout.addWidget(QLabel('Champ de vision'), 1, 0)
        self.fov_slider = QSlider(Qt.Horizontal)
        self.fov_slider.setRange(30, 90)
        self.fov_slider.setValue(45)
        controls_layout.addWidget(self.fov_slider, 1, 1)

        controls_layout.addWidget(QLabel('Vitesse'), 2, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(2)
        controls_layout.addWidget(self.speed_slider, 2, 1)

        controls_layout.addWidget(QLabel('Spot'), 3, 0)
        self.spot_slider = QSlider(Qt.Horizontal)
        self.spot_slider.setRange(1, 90)
        self.spot_slider.setValue(15)
        controls_layout.addWidget(self.spot_slider, 3, 1)

        self.controls.setLayout(controls_layout)
        dock = QDockWidget('Contrôles')
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
            self.glWidget.fixed_spot = not self.glWidget.fixed_spot
            if not self.glWidget.fixed_spot:
                self.glWidget.world.new_fixed_spot = True
        self.glWidget.world.action = action or None

    def mouseReleaseEvent(self, QMouseEvent):
        self.glWidget.world.action = None

    def close(self):
        QtGui.qApp.quit()
