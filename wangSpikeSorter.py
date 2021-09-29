from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMainWindow
from pyqtgraph import PlotWidget, GraphicsLayoutWidget, plot
import pyqtgraph as pg
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.path as mplPath
from intersect import intersection
import glob
import os

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
        self.group_Methods.setGeometry(QtCore.QRect(10, 440, 761, 211))
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
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1410, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
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
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

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
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionLoadFolder.setText(_translate("MainWindow", "Load folder"))
        self.actionLoadFolder.setStatusTip(_translate("MainWindow", "Load a folder"))
        self.actionLoadFolder.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionUndo.setText(_translate("MainWindow", "Undo"))
        self.actionUndo.setStatusTip(_translate("MainWindow", "Undo"))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.actionRedo.setText(_translate("MainWindow", "Redo"))
        self.actionRedo.setStatusTip(_translate("MainWindow", "Redo"))
        self.actionRedo.setShortcut(_translate("MainWindow", "Ctrl+Shift+Z"))

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
        self.pushButton_reset.clicked.connect(self.sw_reset)
        self.pushButton_Add.clicked.connect(self.sw_addpoint)
        self.pushButton_Remove.clicked.connect(self.sw_removepoint)
        self.pushButton_Confirm.clicked.connect(self.sw_confirm)
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
        self.graphicsView_pca.setMenuEnabled(False)
        self.graphicsView_raw.setMenuEnabled(False)
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
                self.units_axes.append(te)
        self.units_axes = np.reshape(self.units_axes, (-1,self.n_maxunit))
    def setup_reset(self):
        self.comboBox_PC1.setCurrentIndex(0)
        self.comboBox_PC2.setCurrentIndex(1)
        self.sw_combobox_pc()
        self.is_addpoint = 0
        self.idx_selected = []
        self.idx_selected_temp = []
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
        self.unit_now = 0
        self.history_units = []
        self.redo_units = []
        self.is_addhistory = True
        self.is_locked = np.zeros(self.n_maxunit) == 1
        cursor = QtCore.Qt.ArrowCursor
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
        for i in range(self.n_maxunit):
            tid = (units == i).squeeze()
            if np.any(tid):
                av[i,] = np.mean(waves[tid,], axis = 0)
        self.av_waves = av
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
        self.update_unit(units)
    def plt_all(self):
        self.plt_pca()
        self.plt_raw()
        self.plt_selectiontool()
        self.plt_units()
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
    def plt_raw(self):
        idp = self.get_selected()
        waves = self.data['waves'].item().copy()
        units = self.data['units'].item().copy()
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
    def plt_units(self):
        waves = self.data['waves'].item().copy()
        units = self.data['units'].item().copy()
        for i in range(self.n_maxunit):
            if i == self.unit_now:
                self.units_axes[0, i].getViewBox().setBackgroundColor("m")
            else:
                self.units_axes[0, i].getViewBox().setBackgroundColor("w")
            if self.is_locked[i]:
                self.units_axes[0, i].showGrid(y = True)
            else:
                self.units_axes[0, i].showGrid(y = False)
            if np.any(units == i):
                tid = (units == i).squeeze()
                lines = MultiLine()
                lines.mysetData(waves[tid,])
                lines.setcolor(self.color_unit[i])
                self.units_axes[0, i].clear()
                self.units_axes[0, i].addItem(lines)
            else:
                self.units_axes[0, i].clear()
    def keyPressEvent(self, event):
        key = event.key()
        if (key == 16777249) | (key == 16777248):
            print('ctr command pressed')
            return;
        str = chr(key)
        if str.isdigit():
            keyint = np.int64(str)
            if (keyint >=0) & (keyint <self.n_maxunit):
                self.unit_now = keyint
                self.plt_all()
        if str.isalpha():
            if str == 'L':
                self.is_locked[self.unit_now] = ~self.is_locked[self.unit_now]
                self.plt_units()
    def get_lockedlines(self):
        units = self.data['units'].item()
        out = np.zeros_like(units)
        for i in range(self.n_maxunit):
            if self.is_locked[i]:
                out[units == i] = 1
        return(out == 1)
    def update_unit(self, units):
        # print('call update_unit')
        idx = self.get_lockedlines()
        units[idx] = self.data['units'].item()[idx]
        if self.is_addhistory:
            # print('add units +1')
            self.history_units.append(self.data['units'].item())
            # print(self.history_units)
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
        cursor = QtCore.Qt.CrossCursor # QCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)
        self.idx_selected_temp = []
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
        self.is_addpoint = 1
    def sw_removepoint(self):
        cursor = QtCore.Qt.IBeamCursor # QCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)
        self.idx_selected_temp = []
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
        self.is_addpoint = -1
    def sw_confirm(self):
        self.update_selected()
        idp = self.idx_selected
        self.is_addpoint = 0
        self.idx_selected = []
        self.idx_selected_temp = []
        self.raw_line_vertices = []
        self.pca_polygon_vertices = []
        cursor = QtCore.Qt.ArrowCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)
        # units = self.data['units'].item()
        if (len(idp) > 0) & np.any(idp):
            # self.is_addhistory = False
            self.update_selectedunit(idp, self.unit_now)
            # self.is_addhistory = True
        # self.update_unit(units)
    def update_selected(self):
        idl = self.get_lockedlines()
        tmp = self.idx_selected_temp
        if (len(tmp) > 0):
            if len(self.idx_selected) > 0:
                if self.is_addpoint == 1:
                    self.idx_selected = (self.idx_selected | tmp) & ~idl
                else:
                    self.idx_selected = (self.idx_selected & ~tmp) & ~idl
            else:
                if self.is_addpoint == 1:
                    self.idx_selected = tmp & ~idl
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
                self.is_addpoint = 0
    def mouse_clicked_pca(self, event):
        if self.is_addpoint != 0:
            p = self.graphicsView_pca.plotItem.vb.mapSceneToView(event.scenePos())
            self.pca_polygon_vertices.append([p.x(), p.y()])
            pts = np.reshape(self.pca_polygon_vertices, [-1, 2])
            pts = np.append(pts, [pts[0,]], axis=0)
            self.assist_addpointsinpolygon(pts)
            if event.button() == 2:
                self.update_selected()
                self.is_addpoint = 0
    def assist_addpointsinpolygon(self, pts):
        poly_path = mplPath.Path(pts)
        pc = self.pc_now
        idp = poly_path.contains_points(pc)
        self.idx_selected_temp = idp
        self.plt_all()
    def assist_addpointsinline(self, pts):
        waves = self.data['waves'].item().copy()
        nl = waves.shape[0]
        npixel = waves.shape[1]
        idp = np.repeat(False, nl)
        idxs = self.idx_selected
        for i in range(nl):
            # if (self.is_addpoint == 1) & (len(idxs)>0) & idxs[i]:
            #     idp[i] = True
            # else:
            #     if (self.is_addpoint == -1) & (len(idxs)>0) & ~idxs[i]:
            #         idp[i] = True
            #     else:
            te = intersection(pts[:, 0], pts[:, 1], list(range(npixel)),waves[i,])
            idp[i] = (te[0].shape[0] > 0)
        self.idx_selected_temp = idp
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
            self.redo_units = self.data['units'].item().copy()
            self.history_units.pop()
            self.is_addhistory = False
            self.update_unit(units)
            self.is_addhistory = True
    def sw_redo(self):
        if len(self.redo_units) > 0:
            self.update_unit(self.redo_units)
            self.redo_units = []

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = SW_MainWindow()
    ui.show()
    # ui.folderName = './'
    # ui.load_folder()
    sys.exit(app.exec_())
