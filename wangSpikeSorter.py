from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMainWindow, QDialog, QPushButton, QComboBox
from pyqtgraph import PlotWidget, GraphicsLayoutWidget, plot
import pyqtgraph as pg
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.path as mplPath
from intersect import intersection
import glob
import os

class Dialog_CombineChannel(QDialog):  # Inheritance of the QDialog class
    def __init__(self, parent = None):
        super(Dialog_CombineChannel, self).__init__(parent)
        self.initUI()
    def setupComboBox(self, nmax):
        self.c1.addItems([str(x) for x in range(nmax)])
        self.c1.setCurrentIndex(1)
        self.c2.addItems([str(x) for x in range(nmax)])
        self.c2.setCurrentIndex(2)
    def initUI(self):
        self.setWindowTitle("Combine channels")  # Window Title
        self.setGeometry(400, 400, 200, 200)
        self.c1 = QComboBox()  # Create a drop-down list box
        self.c2 = QComboBox()
        # for g in get_games():  # Add a selection to the drop-down list box (retrieved from a database query)
        # self.game_item.addItem(g.name, g.id)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)  # window to create confirmation and cancellation buttons
        self.glayout = QtWidgets.QGridLayout()
        self.glayout.addWidget(self.c1, 0, 0)
        self.glayout.addWidget(self.c2, 1, 0)
        self.glayout.addWidget(self.buttons, 1, 1)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.accepted.connect(self.done_choose)
        self.rejected.connect(self.cancel_choose)
        self.setLayout(self.glayout)
    def done_choose(self):
        self.choice = 1
    def cancel_choose(self):
        self.choice = 0
    def getInfo(self):  # Defines the method of obtaining user input data
        return  self.choice, self.c1.currentText(), self.c2.currentText()

class MultiLine(pg.QtGui.QGraphicsPathItem):
    def __init__(self):
        self.path = QtGui.QPainterPath()
        pg.QtGui.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen('k'))
    def mysetData(self, y = [], x = []):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        if (len(y) > 0):
            if (len(x) == 0):
                x = np.empty(y.shape)
                x[:] = np.arange(y.shape[1])[np.newaxis, :]
            connect = np.ones(y.shape, dtype=bool)
            connect[:, -1] = 0  # don't draw the segment between each trace
            x = x.flatten()
            y = y.flatten()
            connect = connect.flatten()
            self.path = pg.arrayToQPath(x, y, connect)
            self.setPath(self.path)
        else:
            self.path = QtGui.QPainterPath()
            self.setPath(self.path)

    def setcolor(self, col):
        self.setPen(pg.mkPen(col))
    def shape(self): # override because QGraphicsPathItem.shape is too expensive.
        return pg.QtGui.QGraphicsItem.shape(self)
    def boundingRect(self):
        return self.path.boundingRect()
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1410, 702)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.group_Units = QtWidgets.QGroupBox(self.centralwidget)
        self.group_Units.setGeometry(QtCore.QRect(780, 0, 621, 591))
        self.group_Units.setObjectName("group_Units")
        self.graphicsView_units = GraphicsLayoutWidget(self.group_Units)
        self.graphicsView_units.setGeometry(QtCore.QRect(10, 30, 601, 551))
        self.graphicsView_units.setObjectName("graphicsView_units")
        self.group_PCA = QtWidgets.QGroupBox(self.centralwidget)
        self.group_PCA.setGeometry(QtCore.QRect(10, 10, 761, 431))
        self.group_PCA.setObjectName("group_PCA")
        self.comboBox_PC1 = QtWidgets.QComboBox(self.group_PCA)
        self.comboBox_PC1.setGeometry(QtCore.QRect(10, 380, 71, 41))
        self.comboBox_PC1.setObjectName("comboBox_PC1")
        self.comboBox_PC1.addItem("")
        self.comboBox_PC1.addItem("")
        self.comboBox_PC1.addItem("")
        self.comboBox_PC1.addItem("")
        self.comboBox_PC2 = QtWidgets.QComboBox(self.group_PCA)
        self.comboBox_PC2.setGeometry(QtCore.QRect(90, 380, 71, 41))
        self.comboBox_PC2.setObjectName("comboBox_PC2")
        self.comboBox_PC2.addItem("")
        self.comboBox_PC2.addItem("")
        self.comboBox_PC2.addItem("")
        self.comboBox_PC2.addItem("")
        self.pushButton_Add = QtWidgets.QPushButton(self.group_PCA)
        self.pushButton_Add.setGeometry(QtCore.QRect(250, 390, 113, 32))
        self.pushButton_Add.setObjectName("pushButton_Add")
        self.pushButton_Remove = QtWidgets.QPushButton(self.group_PCA)
        self.pushButton_Remove.setGeometry(QtCore.QRect(360, 390, 113, 32))
        self.pushButton_Remove.setObjectName("pushButton_Remove")
        self.pushButton_noise = QtWidgets.QPushButton(self.group_PCA)
        self.pushButton_noise.setGeometry(QtCore.QRect(530, 390, 113, 32))
        self.pushButton_noise.setObjectName("pushButton_noise")
        self.pushButton_Confirm = QtWidgets.QPushButton(self.group_PCA)
        self.pushButton_Confirm.setGeometry(QtCore.QRect(640, 390, 113, 32))
        self.pushButton_Confirm.setObjectName("pushButton_Confirm")
        # self.checkBox_deselect = QtWidgets.QCheckBox(self.group_PCA)
        # self.checkBox_deselect.setGeometry(QtCore.QRect(490, 380, 87, 20))
        # self.checkBox_deselect.setObjectName("checkBox_deselect")
        # self.checkBox_useasmodel = QtWidgets.QCheckBox(self.group_PCA)
        # self.checkBox_useasmodel.setGeometry(QtCore.QRect(490, 400, 111, 20))
        # self.checkBox_useasmodel.setChecked(True)
        # self.checkBox_useasmodel.setObjectName("checkBox_useasmodel")
        self.graphicsView_pca = PlotWidget(self.group_PCA)
        self.graphicsView_pca.setGeometry(QtCore.QRect(10, 30, 351, 351))
        self.graphicsView_pca.setObjectName("graphicsView_pca")
        self.graphicsView_raw = PlotWidget(self.group_PCA)
        self.graphicsView_raw.setGeometry(QtCore.QRect(370, 30, 381, 351))
        self.graphicsView_raw.setObjectName("graphicsView_raw")
        self.pushButton_reset = QtWidgets.QPushButton(self.group_PCA)
        self.pushButton_reset.setGeometry(QtCore.QRect(170, 390, 81, 32))
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.group_Methods = QtWidgets.QGroupBox(self.centralwidget)
        self.group_Methods.setGeometry(QtCore.QRect(10, 440, 261, 211))
        self.group_Methods.setObjectName("group_Methods")
        self.comboBox_ClusterMethods = QtWidgets.QComboBox(self.group_Methods)
        self.comboBox_ClusterMethods.setGeometry(QtCore.QRect(10, 40, 204, 26))
        self.comboBox_ClusterMethods.setObjectName("comboBox_ClusterMethods")
        self.comboBox_ClusterMethods.addItem("")
        # self.pushButton_sortsafe = QtWidgets.QPushButton(self.group_Methods)
        # self.pushButton_sortsafe.setGeometry(QtCore.QRect(10, 140, 171, 32))
        # self.pushButton_sortsafe.setObjectName("pushButton_sortsafe")
        self.pushButton_sortall = QtWidgets.QPushButton(self.group_Methods)
        self.pushButton_sortall.setGeometry(QtCore.QRect(10, 170, 113, 32))
        self.pushButton_sortall.setObjectName("pushButton_sortall")
        # self.textEdit_sortsafe = QtWidgets.QTextEdit(self.group_Methods)
        # self.textEdit_sortsafe.setGeometry(QtCore.QRect(180, 140, 71, 31))
        # self.textEdit_sortsafe.setObjectName("textEdit_sortsafe")
        self.frame_Channel = QtWidgets.QFrame(self.centralwidget)
        self.frame_Channel.setGeometry(QtCore.QRect(780, 600, 621, 51))
        self.frame_Channel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Channel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Channel.setObjectName("frame_Channel")
        self.label_channel = QtWidgets.QLabel(self.frame_Channel)
        self.label_channel.setGeometry(QtCore.QRect(10, 10, 171, 31))
        self.label_channel.setObjectName("label_channel")
        # self.textEdit_channel = QtWidgets.QTextEdit(self.frame_Channel)
        # self.textEdit_channel.setGeometry(QtCore.QRect(190, 10, 81, 31))
        # self.textEdit_channel.setObjectName("textEdit_channel")
        self.comboBox_channel = QtWidgets.QComboBox(self.frame_Channel)
        self.comboBox_channel.setGeometry(QtCore.QRect(310, 10, 81, 31))
        self.comboBox_channel.setObjectName("comboBox_channel")
        # self.pushButton_gotochannel = QtWidgets.QPushButton(self.frame_Channel)
        # self.pushButton_gotochannel.setGeometry(QtCore.QRect(280, 10, 113, 32))
        # self.pushButton_gotochannel.setObjectName("pushButton_gotochannel")
        self.pushButton_previouschannel = QtWidgets.QPushButton(self.frame_Channel)
        self.pushButton_previouschannel.setGeometry(QtCore.QRect(390, 10, 113, 32))
        self.pushButton_previouschannel.setObjectName("pushButton_previouschannel")
        self.pushButton_nextchannel = QtWidgets.QPushButton(self.frame_Channel)
        self.pushButton_nextchannel.setGeometry(QtCore.QRect(500, 10, 113, 32))
        self.pushButton_nextchannel.setObjectName("pushButton_nextchannel")
        self.groupBox_side = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_side.setGeometry(QtCore.QRect(280, 440, 491, 211))
        self.groupBox_side.setObjectName("groupBox_side")
        self.graphicsView_side1 = PlotWidget(self.groupBox_side)
        self.graphicsView_side1.setGeometry(QtCore.QRect(0, 20, 241, 191))
        self.graphicsView_side1.setObjectName("graphicsView_side1")
        self.graphicsView_side2 = PlotWidget(self.groupBox_side)
        self.graphicsView_side2.setGeometry(QtCore.QRect(240, 21, 251, 191))
        self.graphicsView_side2.setObjectName("graphicsView_side2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1410, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuFunction = QtWidgets.QMenu(self.menubar)
        self.menuFunction.setObjectName("menuFunction")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoadFolder = QtWidgets.QAction(MainWindow)
        self.actionLoadFolder.setObjectName("actionLoadFolder")
        self.menuFile.addAction(self.actionLoadFolder)
        self.actionUndo = QtWidgets.QAction(MainWindow)
        self.actionUndo.setObjectName("actionUndo")
        self.menuEdit.addAction(self.actionUndo)
        self.actionRedo = QtWidgets.QAction(MainWindow)
        self.actionRedo.setObjectName("actionRedo")
        self.menuEdit.addAction(self.actionRedo)
        self.RemoveChannel = QtWidgets.QAction(MainWindow)
        self.RemoveChannel.setObjectName("RemoveChannel")
        self.menuFunction.addAction(self.RemoveChannel)
        self.CombineChannels = QtWidgets.QAction(MainWindow)
        self.CombineChannels.setObjectName("CombineChannels")
        self.menuFunction.addAction(self.CombineChannels)
        self.SqueezeChannels = QtWidgets.QAction(MainWindow)
        self.SqueezeChannels.setObjectName("SqueezeChannels")
        self.menuFunction.addAction(self.SqueezeChannels)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuFunction.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.group_Units.setTitle(_translate("MainWindow", "Neurons"))
        self.group_PCA.setTitle(_translate("MainWindow", "PCA and Raw Signals"))
        self.comboBox_PC1.setItemText(0, _translate("MainWindow", "PC1"))
        self.comboBox_PC1.setItemText(1, _translate("MainWindow", "PC2"))
        self.comboBox_PC1.setItemText(2, _translate("MainWindow", "PC3"))
        self.comboBox_PC1.setItemText(3, _translate("MainWindow", "PC4"))
        self.comboBox_PC2.setItemText(0, _translate("MainWindow", "PC1"))
        self.comboBox_PC2.setItemText(1, _translate("MainWindow", "PC2"))
        self.comboBox_PC2.setItemText(2, _translate("MainWindow", "PC3"))
        self.comboBox_PC2.setItemText(3, _translate("MainWindow", "PC4"))
        self.pushButton_Add.setText(_translate("MainWindow", "add point"))
        self.pushButton_Remove.setText(_translate("MainWindow", "remove point"))
        self.pushButton_Confirm.setText(_translate("MainWindow", "Confirm"))
        self.pushButton_noise.setText(_translate("MainWindow", "Noise"))
        # self.checkBox_deselect.setText(_translate("MainWindow", "de-select"))
        # self.checkBox_useasmodel.setText(_translate("MainWindow", "use as model"))
        self.pushButton_reset.setText(_translate("MainWindow", "Reset"))
        self.group_Methods.setTitle(_translate("MainWindow", "Clustering"))
        # self.pushButton_sortsafe.setText(_translate("MainWindow", "Cluster with confidence"))
        self.pushButton_sortall.setText(_translate("MainWindow", "Cluster all"))
        self.comboBox_ClusterMethods.setItemText(0, _translate("MainWindow", "minimal distance"))
        self.label_channel.setText(_translate("MainWindow", "Load data first"))
        # self.pushButton_gotochannel.setText(_translate("MainWindow", "Go to Channel"))
        self.pushButton_previouschannel.setText(_translate("MainWindow", "Previous"))
        self.pushButton_nextchannel.setText(_translate("MainWindow", "Next Channel"))
        self.groupBox_side.setTitle(_translate("MainWindow", "Side Plots"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuFunction.setTitle(_translate("MainWindow", "Function"))
        self.actionLoadFolder.setText(_translate("MainWindow", "Load folder"))
        self.actionLoadFolder.setStatusTip(_translate("MainWindow", "Load a folder"))
        self.actionLoadFolder.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionUndo.setText(_translate("MainWindow", "Undo"))
        self.actionUndo.setStatusTip(_translate("MainWindow", "Undo"))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.actionRedo.setText(_translate("MainWindow", "Redo"))
        self.actionRedo.setStatusTip(_translate("MainWindow", "Redo"))
        self.actionRedo.setShortcut(_translate("MainWindow", "Ctrl+Shift+Z"))
        self.RemoveChannel.setText(_translate("MainWindow", "Remove Channel"))
        self.CombineChannels.setText(_translate("MainWindow", "Combine Channels"))
        self.SqueezeChannels.setText(_translate("MainWindow", "Squeeze Channels"))

class SW_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent = parent)
        self.setupUi(self)
        self.setup_SW()
    def setup_SW(self):
        # set up colors
        self.color_unit = ["k","r","b","g","c","y"]
        self.n_maxunit = len(self.color_unit)
        self.is_loaddata = False
        self.setup_reset()
        self.setup_axes()
        self.setup_connect()
    def setup_connect(self):
        self.actionLoadFolder.triggered.connect(self.sw_load_folder)
        self.actionUndo.triggered.connect(self.sw_undo)
        self.actionRedo.triggered.connect(self.sw_redo)
        self.RemoveChannel.triggered.connect(self.sw_removechannel)
        self.CombineChannels.triggered.connect(self.sw_combinechannels)
        self.SqueezeChannels.triggered.connect(self.sw_squeezechannels)
        self.pushButton_reset.clicked.connect(self.sw_reset)
        self.pushButton_Add.clicked.connect(self.sw_addpoint)
        self.pushButton_Remove.clicked.connect(self.sw_removepoint)
        self.pushButton_Confirm.clicked.connect(self.sw_confirm)
        self.pushButton_noise.clicked.connect(self.sw_noise)
        self.pushButton_nextchannel.clicked.connect(self.sw_nextchannel)
        self.pushButton_previouschannel.clicked.connect(self.sw_previouschannel)
        self.pushButton_sortall.clicked.connect(self.sw_sortall)
        self.pca_emptyplot.scene().sigMouseClicked.connect(self.mouse_clicked_pca)
        self.raw_emptyplot.scene().sigMouseClicked.connect(self.mouse_clicked_raw)
        self.comboBox_PC1.activated.connect(self.sw_combobox_pc)
        self.comboBox_PC2.activated.connect(self.sw_combobox_pc)
        self.comboBox_channel.activated.connect(self.sw_gotochannel)
    def setup_axes(self):
        # set up graphics view background
        self.graphicsView_pca.setBackground('w')
        self.graphicsView_raw.setBackground('w')
        self.graphicsView_units.setBackground('w')
        self.graphicsView_side1.setBackground('w')
        self.graphicsView_side2.setBackground('w')
        self.graphicsView_pca.setMenuEnabled(False)
        self.graphicsView_raw.setMenuEnabled(False)
        self.graphicsView_side1.setMenuEnabled(False)
        self.graphicsView_side2.setMenuEnabled(False)
        # set up lines
        # -- raw
        self.raw_emptyplot = self.graphicsView_raw.plot(x=[], y=[], pen=pg.mkPen("m"))
        self.raw_lines = []
        for ui in range(self.n_maxunit):
            lines = MultiLine()
            lines.setcolor(self.color_unit[ui])
            self.raw_lines.append(lines)
            self.graphicsView_raw.addItem(lines)
        lines = MultiLine()
        lines.setcolor("m")
        self.lines_selected = lines
        self.graphicsView_raw.addItem(lines)
        te = pg.PlotCurveItem(pen=pg.mkPen("m"))  # this color needs to be changed
        self.raw_path = te
        self.graphicsView_raw.addItem(te)
        # -- pca
        self.pca_emptyplot = self.graphicsView_pca.plot(x=[], y=[], pen=pg.mkPen("m"))
        self.pca_scatter = []
        for ui in range(self.n_maxunit):
            te = pg.ScatterPlotItem(brush=pg.mkBrush(self.color_unit[ui]))
            te.setSize(2)
            self.pca_scatter.append(te)
            self.graphicsView_pca.addItem(te)
        te = pg.ScatterPlotItem(brush=pg.mkBrush("m"))
        te.setSize(5)
        self.points_selected = te
        self.graphicsView_pca.addItem(te)
        te = pg.PlotCurveItem(pen=pg.mkPen("m"))  # this color needs to be changed
        self.pca_path = te
        self.graphicsView_pca.addItem(te)
        # -- units
        self.units_axes = []
        for uj in range(5):
            for ui in range(self.n_maxunit):
                te = self.graphicsView_units.addPlot(row = uj, col = ui)
                # te.setAspectLocked(lock=False)
                # self.graphicsView_units.addViewBox(row = uj, col = ui)
                self.units_axes.append(te)
        self.units_axes = np.reshape(self.units_axes, (-1,self.n_maxunit))
    def setup_reset(self):
        self.comboBox_PC1.setCurrentIndex(0)
        self.comboBox_PC2.setCurrentIndex(1)
        self.sw_combobox_pc()
        self.set_addpoint(0)
        self.idx_selected = []
        self.idx_selected_temp = []
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
        self.unit_now = 0
        self.history_units = []
        self.history_locked = []
        self.redo_units = []
        self.redo_locked = []
        self.is_addhistory = True
        self.is_locked = np.zeros(self.n_maxunit) == 1
    def set_addpoint(self, pt):
        self.is_addpoint = pt
        if pt == 0:
            cursor = QtCore.Qt.ArrowCursor
        else:
            if pt == 1:
                cursor = QtCore.Qt.CrossCursor  # QCursor
            else:
                cursor = QtCore.Qt.IBeamCursor # QCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)
    def file_loadfile(self):
        self.setup_reset()
        self.is_loaddata = True
        filename = self.filenow
        td = sio.loadmat(filename)
        self.rawmat = td
        self.data = td.get('waveforms')
        self.initial_dataformat()
        self.comp_setup()
        self.statusbar.showMessage(f"loaded file: {filename}")
    def initial_dataformat(self):
        units = self.data['units'].item().copy()
        if (len(units) == 1):
            units = units[0]
            self.data['units'].itemset(units)
        waves = self.data['waves'].item().copy()
        self.data['waves'].itemset(waves.T)
    def comp_setup(self):
        # compute PCA
        waves = self.data['waves'].item().copy()
        self.pca = self.PCA(waves)
        self.sw_combobox_pc()
        self.comp_default()
        self.plt_all()
    def comp_default(self):
        # pc = self.pca
        units = self.data['units'].item().copy()
        waves = self.data['waves'].item().copy()
        npix = waves.shape[1]
        av = np.zeros((self.n_maxunit, npix))
        sd = np.zeros((self.n_maxunit, npix))
        for i in range(self.n_maxunit):
            tid = (units == i).squeeze()
            if np.any(tid):
                av[i,] = np.mean(waves[tid,], axis = 0)
                sd[i,] = np.std(waves[tid,], axis = 0)#/np.sqrt(np.sum(tid))
        self.av_waves = av
        self.sd_waves = sd
        dist = np.zeros((waves.shape[0], self.n_maxunit))
        for i in range(self.n_maxunit):
            dist[:,i] = np.mean((waves - av[i,])**2, axis = 1)
        self.dist_waves = dist
    def PCA(self, X, num_components=[]):
        if len(num_components) == 0:
            num_components = X.shape[1]
        # Step-1
        X_meaned = X - np.mean(X, axis=0)

        # Step-2
        cov_mat = np.cov(X_meaned, rowvar=False)

        # Step-3
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        # Step-4
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        # Step-5
        eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

        # Step-6
        X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

        return X_reduced
    def sw_combobox_pc(self):
        if self.is_loaddata:
            n1 = self.comboBox_PC1.currentText()
            n1 = int(n1[2]) - 1
            n2 = self.comboBox_PC2.currentText()
            n2 = int(n2[2]) - 1
            pc = self.pca[:, (n1, n2)]
            self.pc_now = pc
            self.pca_polygon_vertices = []
            self.idx_selected_temp = []
            self.plt_pca()
    def sw_reset(self):
        # self.setup_reset()
        units = self.data['units'].item().copy()
        units = np.zeros(units.shape)
        units = np.int64(units)
        islocked = np.zeros_like(self.is_locked) == 1
        self.update_unit(units, islocked)
    def plt_all(self):
        self.plt_pca()
        self.plt_raw()
        self.plt_selectiontool()
        self.plt_units()
        self.plt_locked()
        self.plt_noise()
    def plt_noise(self):
        waves = self.data['waves'].item().copy()
        units = self.data['units'].item().copy()

        self.graphicsView_side1.clear()
        self.graphicsView_side1.setLabel('left', 'Voltage')
        self.graphicsView_side1.setLabel('bottom', 'Time')
        for i in range(self.n_maxunit):
            # special plot - average
            cv1 = pg.PlotCurveItem(self.av_waves[i,] + self.sd_waves[i,])
            cv2 = pg.PlotCurveItem(self.av_waves[i,] - self.sd_waves[i,])
            tl = pg.FillBetweenItem(curve1=cv1, curve2=cv2, brush=pg.mkBrush(self.color_unit[i]))
            # tl = pg.PlotCurveItem(self.av_waves[i,], fill = -0.3, ,  pen=pg.mkPen(self.color_unit[i]))
            self.graphicsView_side1.addItem(tl)
            str = f"sorted: {np.mean(units > 0) * 100:.2f}%, {np.sum(units > 0)}/{len(units)}"
            self.graphicsView_side1.setTitle(str)

        self.graphicsView_side2.clear()
        str = f"unsorted: {np.mean(units == -1)*100:.2f}%, {np.sum(units == -1)}/{len(units)}"
        self.graphicsView_side2.setTitle(str)
        self.graphicsView_side2.setLabel('left', 'Voltage')
        self.graphicsView_side2.setLabel('bottom', 'Time')
        tid = (units == -1).squeeze()
        if (any(tid)):
            tl = MultiLine()
            tl.mysetData(waves[tid,])
            tl.setcolor(pg.mkPen('k'))
            self.graphicsView_side2.addItem(tl)

    def plt_pca(self):
        units = self.data['units'].item().copy()
        pc = self.pc_now
        idp = self.get_selected()
        for ui in range(self.n_maxunit):
            tid = (units == ui).squeeze()
            if len(idp) > 0:
                tid = tid & ~idp
            if np.any(tid):
                self.pca_scatter[ui].setData(x = pc[tid,0], y = pc[tid,1])
            else:
                self.pca_scatter[ui].setData(x=[], y=[])
        if (len(idp) > 0) & np.any(idp):
            self.points_selected.setData(x=pc[idp, 0], y=pc[idp, 1])
        else:
            self.points_selected.setData(x=[], y=[])

        self.graphicsView_pca.setLabel('bottom',f'{self.comboBox_PC1.currentText()}')
        self.graphicsView_pca.setLabel('left',f'{self.comboBox_PC2.currentText()}')
    def plt_raw(self):
        idp = self.get_selected()
        waves = self.data['waves'].item().copy()
        units = self.data['units'].item().copy()
        self.graphicsView_raw.setLabel('left','Voltage')
        self.graphicsView_raw.setLabel('bottom','Time')
        for ui in range(self.n_maxunit):
            tid = (units == ui).squeeze()
            if len(idp) > 0:
                tid = tid & ~idp
            if np.any(tid):
                self.raw_lines[ui].mysetData(waves[tid,])
            else:
                self.raw_lines[ui].mysetData()
        if (len(idp) > 0) & np.any(idp):
            self.lines_selected.mysetData(waves[idp,])
        else:
            self.lines_selected.mysetData()
    def plt_selectiontool(self):
        if len(self.pca_polygon_vertices) > 0:
            pts = np.reshape(self.pca_polygon_vertices, [-1, 2])
            pts = np.append(pts, [pts[0,]], axis=0)
            self.pca_path.setData(x=pts[:, 0], y=pts[:, 1])
        else:
            self.pca_path.setData(x = [], y = [])
        if len(self.raw_line_vertices) > 0:
            pts = np.reshape(self.raw_line_vertices, [-1, 2])
            pts = np.append(pts, [pts[0,]], axis=0)
            self.raw_path.setData(x=pts[:, 0], y=pts[:, 1])
        else:
            self.raw_path.setData(x = [], y = [])
    def plt_locked(self):
        for i in range(self.n_maxunit):
            if self.is_locked[i]:
                str_L = 'Locked'
                self.units_axes[1,i].setTitle(str_L)
            else:
                str_L = ''
                self.units_axes[1,i].setTitle(str_L)
    def plt_units(self):
        waves = self.data['waves'].item().copy()
        units = self.data['units'].item().copy()
        trg = np.array([np.infty, -np.infty])
        for i in range(self.n_maxunit):
            if i == self.unit_now:
                self.units_axes[0, i].getViewBox().setBackgroundColor("m")
            else:
                self.units_axes[0, i].getViewBox().setBackgroundColor("w")
            n_uniti = np.sum(units == i)
            n_unitall = len(units)
            str = f"{n_uniti/n_unitall*100:.1f}%"
            # str = str + str_L
            self.units_axes[0, i].setTitle(str) # fake title
            self.units_axes[0, i].clear()
            self.units_axes[1, i].clear()
            self.units_axes[2, i].clear()
            self.units_axes[3, i].clear()
            if n_uniti > 0:
                tid = (units == i).squeeze()
                # lines
                lines = MultiLine()
                lines.mysetData(waves[tid,])
                lines.setcolor(self.color_unit[i])
                self.units_axes[0, i].addItem(lines)
                self.units_axes[0, i].autoRange()
                te = self.units_axes[0, i].getAxis('left').range
                if te[1] > trg[1]:
                    trg[1] = te[1]
                if te[0] < trg[0]:
                    trg[0] = te[0]
                # ITI
                st = self.data['spikeTimes'].item().squeeze()
                hst_y, hst_x = np.histogram(np.diff(st[tid]), bins=np.linspace(0, 100, 20))
                thst = pg.PlotCurveItem(hst_x, hst_y, stepMode=True, fillLevel=0, brush=pg.mkBrush(self.color_unit[i]))
                self.units_axes[1, i].addItem(thst)
                self.units_axes[1, i].autoRange()
                # timing vs firing rate
                ty, tx = np.histogram(st[tid]/np.max(st), bins=np.linspace(0, 1, 100))
                tx = (tx[1:] + tx[:-1]) / 2
                thst = pg.PlotCurveItem(tx, ty, pen=pg.mkPen(self.color_unit[i]))
                self.units_axes[2, i].addItem(thst)
                self.units_axes[2, i].autoRange()
                # distance from clusters
                dst = self.dist_waves
                bin = np.linspace(0, np.mean(dst[tid,i]) * 2, 100)
                ldsts = np.zeros((self.n_maxunit, len(bin)-1))
                for j in range(self.n_maxunit):
                    ty, tx = np.histogram(dst[units == j, i], bins = bin)
                    tx = (tx[1:] + tx[:-1])/2
                    tl = pg.PlotCurveItem(tx, ty, pen=pg.mkPen(self.color_unit[j]))
                    self.units_axes[3, i].addItem(tl)
                self.units_axes[3, i].autoRange()
        for i in range(self.n_maxunit):
            self.units_axes[0, i].setYRange(trg[0], trg[1])
    def keyPressEvent(self, event):
        key = event.key()
        if key < 200: #ascii codes range
            str = chr(key)
            if str.isdigit():
                keyint = np.int64(str)
                if (keyint >=0) & (keyint <self.n_maxunit):
                    self.unit_now = keyint
                    self.plt_all()
            if str.isalpha():
                if str == 'L':
                    self.is_locked[self.unit_now] = ~self.is_locked[self.unit_now]
                    self.plt_locked()
                if str == 'A':
                    self.update_selected()
                    self.set_addpoint(0)
    def get_lockedlines(self):
        units = self.data['units'].item()
        out = np.zeros_like(units)
        for i in range(self.n_maxunit):
            if self.is_locked[i]:
                out[units == i] = 1
        return(out == 1)
    def update_unit(self, units, locked = []):
        if self.is_addhistory:
            self.history_units.append(self.data['units'].item())
            self.history_locked.append(self.is_locked)
        if len(locked) == 0:
            idx = self.get_lockedlines()
            units[idx] = self.data['units'].item()[idx]
        else:
            self.is_locked = locked
        self.data['units'].itemset(units)
        self.comp_default()
        self.plt_all()
        self.autosave()
    def update_selectedunit(self, idx, unitnew):
        units = self.data['units'].item().copy()
        units[idx] = unitnew
        self.update_unit(units)
    def autosave(self):
        # reverse waves
        waves = self.data['waves'].item().copy()
        self.data['waves'].itemset(waves.T)
        mdict = self.rawmat
        mdict['waveforms'] = self.data
        sio.savemat(self.filenow, mdict)
        self.data['waves'].itemset(waves)
    def sw_addpoint(self):
        self.idx_selected_temp = []
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
        self.set_addpoint(1)
    def sw_removepoint(self):
        self.idx_selected_temp = []
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
        self.set_addpoint(-1)
    def sw_confirm(self):
        self.update_selected()
        idp = self.idx_selected
        self.set_addpoint(0)
        self.idx_selected = []
        self.idx_selected_temp = []
        self.raw_line_vertices = []
        self.pca_polygon_vertices = []
        # units = self.data['units'].item()
        if (len(idp) > 0) & np.any(idp):
            # self.is_addhistory = False
            self.update_selectedunit(idp, self.unit_now)
            # self.is_addhistory = True
        # self.update_unit(units)

    def sw_noise(self):
            self.update_selected()
            idp = self.idx_selected
            self.set_addpoint(0)
            self.idx_selected = []
            self.idx_selected_temp = []
            self.raw_line_vertices = []
            self.pca_polygon_vertices = []
            if (len(idp) > 0) & np.any(idp):
                self.update_selectedunit(idp, -1)
    def select_locked(self):
        idl = self.get_lockedlines()
        if len(self.idx_selected) > 0:
            self.idx_selected = self.idx_selected & ~idl
        if len(self.idx_selected_temp) > 0:
            self.idx_selected_temp = self.idx_selected_temp & ~idl

    def update_selected(self):
        tmp = self.idx_selected_temp
        if (len(tmp) > 0):
            if len(self.idx_selected) > 0:
                if self.is_addpoint == 1:
                    self.idx_selected = (self.idx_selected | tmp)
                else:
                    self.idx_selected = (self.idx_selected & ~tmp)
            else:
                if self.is_addpoint == 1:
                    self.idx_selected = tmp
                else:
                    self.idx_selected = []
        self.idx_selected_temp = []
        self.plt_all()
    def get_selected(self):
        idp = self.idx_selected
        if len(self.idx_selected_temp) > 0:
            if len(idp) > 0:
                if self.is_addpoint == 1:
                    idp = idp | self.idx_selected_temp
                else:
                    idp = idp & ~self.idx_selected_temp
            else:
                if self.is_addpoint == 1:
                    idp = self.idx_selected_temp
                else:
                    idp = []
        return(idp)
    def mouse_clicked_raw(self, event):
        if self.is_addpoint != 0:
            p = self.graphicsView_raw.plotItem.vb.mapSceneToView(event.scenePos())
            self.raw_line_vertices.append([p.x(), p.y()])
            while len(self.raw_line_vertices) > 2:
                self.raw_line_vertices.reverse()
                self.raw_line_vertices.pop()
                self.raw_line_vertices.reverse()
            if len(self.raw_line_vertices) == 2:
                pts = np.reshape(self.raw_line_vertices, [-1, 2])
                self.assist_addpointsinline(pts)
            if event.button() == 2:
                self.update_selected()
                self.set_addpoint(0)
    def mouse_clicked_pca(self, event):
        if self.is_addpoint != 0:
            p = self.graphicsView_pca.plotItem.vb.mapSceneToView(event.scenePos())
            self.pca_polygon_vertices.append([p.x(), p.y()])
            pts = np.reshape(self.pca_polygon_vertices, [-1, 2])
            pts = np.append(pts, [pts[0,]], axis=0)
            self.assist_addpointsinpolygon(pts)
            if event.button() == 2:
                self.update_selected()
                self.set_addpoint(0)
    def assist_addpointsinpolygon(self, pts):
        poly_path = mplPath.Path(pts)
        pc = self.pc_now
        idp = poly_path.contains_points(pc)
        self.idx_selected_temp = idp
        self.select_locked()
        self.plt_all()
    def assist_addpointsinline(self, pts):
        waves = self.data['waves'].item().copy()
        nl = waves.shape[0]
        npixel = waves.shape[1]
        idp = np.repeat(False, nl)
        # for i in range(nl):
        #     te = intersection(pts[:, 0], pts[:, 1], list(range(npixel)),waves[i,])
        #     idp[i] = (te[0].shape[0] > 0)
        x_coords, y_coords = zip(*pts)
        xx = np.sort(x_coords)
        yy = np.sort(y_coords)
        if np.diff(xx) == 0:
            xx = np.unique(xx)[0]
            if (xx >= 1) & (xx <= waves.shape[1]):
                t1 = int(np.floor(xx))
                t2 = int(np.ceil(xx))
                if t1 == t2:
                    y2 = waves[:, t1 - 1]
                else:
                    tr = (xx - t1) / (t2 - t1)
                    y2 = waves[:, t1] * (1 - tr) + waves[:, t2] * tr
                idp = (y2 >= yy[0]) & (y2 <= yy[1])
            else:
                idp = []
        else:
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = np.linalg.lstsq(A, y_coords, rcond = None)[0]
            xs = range(int(np.floor(xx[1])- np.ceil(xx[0]))) + np.ceil(xx[0])
            if len(xs) == 0:
                xs = xx
            else:
                if xs[0] != xx[0]:
                    xs = np.append(xx[0],xs)
                if xs[-1] != xx[1]:
                    xs = np.append(xs, xx[1])
            xs = xs[(xs >= 1) & (xs <= waves.shape[1])]
            ys = xs * m + c
            if len(xs) > 0:
                y2 = np.zeros((waves.shape[0], len(xs)))
                for i in range(len(xs)):
                    t1 = int(np.floor(xs[i]))
                    t2 = int(np.ceil(xs[i]))
                    if t1 == t2:
                        y2[:, i] = waves[:, t1 - 1]
                    else:
                        tr = (xs[i] - t1)/(t2-t1)
                        y2[:, i] = waves[:, t1] * (1-tr) + waves[:, t2] * tr
                dy = y2 - ys
                idp = np.any(dy > 0, axis=1) & np.any(dy < 0, axis = 1)
        self.idx_selected_temp = idp
        self.select_locked()
        self.plt_all()
    def choosefile(self, fid):
        self.fileid = fid
        self.filenow = os.path.join(self.folderName, self.filelists[fid])
        self.label_channel.setText(f'channel {fid+1} / {self.n_file}')
        self.comboBox_channel.setCurrentIndex(fid)
        # self.textEdit_channel.setText(f'{fid+1}')
        self.file_loadfile()  # import the first file
    def sw_gotochannel(self, fid):
        fnow = self.comboBox_channel.currentText()
        xx = [x in fnow for x in self.filelists]
        fid = np.where(xx)[0][0]
        self.choosefile(fid)
    def load_folder(self):
        fs = os.listdir(self.folderName)
        fs.sort()
        self.filelists = [x for x in fs if x.startswith('waveforms')]
        self.comboBox_channel.clear()
        self.comboBox_channel.addItems(self.filelists)
        self.n_file = len(self.filelists)
        self.choosefile(0)
    def sw_load_folder(self):
        dlg = QFileDialog()
        if dlg.exec_():
            self.folderName = dlg.selectedFiles()[0]
            if self.folderName:
                self.load_folder()
    def sw_previouschannel(self):
        self.fileid = self.fileid - 1
        if self.fileid < 0:
            self.fileid = 0
        self.choosefile(self.fileid)
    def sw_nextchannel(self):
        self.fileid = self.fileid + 1
        if self.fileid >= self.n_file:
            self.fileid = self.n_file - 1
        self.choosefile(self.fileid)
    def sw_sortall(self):
        if self.comboBox_ClusterMethods.currentText() == "minimal distance":
            dists = self.dist_waves
            dists_u = dists[:,range(1,self.n_maxunit-1)]
            units_predict = dists_u.argmin(axis = 1) + 1
            self.update_unit(units_predict)
    def sw_undo(self):
        if len(self.history_units) > 0:
            units = self.history_units[-1].copy()
            locked = self.history_locked[-1].copy()
            self.redo_units = self.data['units'].item().copy()
            self.redo_locked = self.is_locked
            self.history_units.pop()
            self.history_locked.pop()
            self.is_addhistory = False
            self.update_unit(units, locked)
            self.is_addhistory = True
    def sw_redo(self):
        if len(self.redo_units) > 0:
            self.update_unit(self.redo_units, self.redo_locked)
            self.redo_units = []
            self.redo_locked = []
    def sw_removechannel(self):
        unow = self.unit_now
        self.is_locked[unow] = False
        units = self.data['units'].item().copy()
        units[units == unow] = 0
        self.update_unit(units)
    def sw_combinechannels(self):
        dlg = Dialog_CombineChannel(self)
        dlg.setupComboBox(self.n_maxunit)
        if dlg.exec():
            c, c1, c2 = dlg.getInfo()
            if c == 1:
                c1 = int(c1)
                c2 = int(c2)
                if c1 != c2:
                    units = self.data['units'].item().copy()
                    units[units == c2] = c1
                    islocked = self.is_locked.copy()
                    islocked[c2] = False
                    self.update_unit(units, islocked)
    def sw_squeezechannels(self):
        units = self.data['units'].item().copy()
        ct = np.zeros(self.n_maxunit)
        for i in range(self.n_maxunit):
            ct[i] = np.sum(units == i)
        us = np.nonzero(ct > 0)[0]
        nu = len(us)
        islocked = self.is_locked.copy()
        if nu < np.max(us)+1:
            for i in range(self.n_maxunit):
                if i < nu:
                    if i != us[i]:
                        units[units == us[i]] = i
                        islocked[i] = islocked[us[i]]
                else:
                    islocked[i] = False
            self.update_unit(units, islocked)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = SW_MainWindow()
    ui.show()
    # ui.folderName = './'
    # ui.load_folder()
    sys.exit(app.exec_())
