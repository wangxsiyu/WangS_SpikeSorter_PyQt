from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMainWindow
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
from pyqtgraph import PlotWidget, GraphicsLayoutWidget, plot
import pyqtgraph as pg
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.path as mplPath
from intersect import intersection

class MultiLine(pg.QtGui.QGraphicsPathItem):
    def __init__(self, y, x = []):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        if len(x) == 0:
            x = np.empty(y.shape)
            x[:] = np.arange(y.shape[1])[np.newaxis, :]
        connect = np.ones(y.shape, dtype=bool)
        connect[:,-1] = 0 # don't draw the segment between each trace
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        pg.QtGui.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen('k'))
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
        self.comboBox_ClusterMethods.setGeometry(QtCore.QRect(10, 40, 104, 26))
        self.comboBox_ClusterMethods.setObjectName("comboBox_ClusterMethods")
        self.pushButton_sortsafe = QtWidgets.QPushButton(self.group_Methods)
        self.pushButton_sortsafe.setGeometry(QtCore.QRect(10, 140, 171, 32))
        self.pushButton_sortsafe.setObjectName("pushButton_sortsafe")
        self.pushButton_sortall = QtWidgets.QPushButton(self.group_Methods)
        self.pushButton_sortall.setGeometry(QtCore.QRect(10, 170, 113, 32))
        self.pushButton_sortall.setObjectName("pushButton_sortall")
        self.textEdit_sortsafe = QtWidgets.QTextEdit(self.group_Methods)
        self.textEdit_sortsafe.setGeometry(QtCore.QRect(180, 140, 71, 31))
        self.textEdit_sortsafe.setObjectName("textEdit_sortsafe")
        self.frame_Channel = QtWidgets.QFrame(self.centralwidget)
        self.frame_Channel.setGeometry(QtCore.QRect(780, 600, 621, 51))
        self.frame_Channel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_Channel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_Channel.setObjectName("frame_Channel")
        self.label_channel = QtWidgets.QLabel(self.frame_Channel)
        self.label_channel.setGeometry(QtCore.QRect(10, 10, 171, 31))
        self.label_channel.setObjectName("label_channel")
        self.textEdit_channel = QtWidgets.QTextEdit(self.frame_Channel)
        self.textEdit_channel.setGeometry(QtCore.QRect(190, 10, 81, 31))
        self.textEdit_channel.setObjectName("textEdit_channel")
        self.pushButton_gotochannel = QtWidgets.QPushButton(self.frame_Channel)
        self.pushButton_gotochannel.setGeometry(QtCore.QRect(280, 10, 113, 32))
        self.pushButton_gotochannel.setObjectName("pushButton_gotochannel")
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
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoadFolder = QtWidgets.QAction(MainWindow)
        self.actionLoadFolder.setObjectName("actionLoadFolder")
        self.menuFile.addAction(self.actionLoadFolder)
        self.menubar.addAction(self.menuFile.menuAction())

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
        self.pushButton_sortsafe.setText(_translate("MainWindow", "Cluster with confidence"))
        self.pushButton_sortall.setText(_translate("MainWindow", "Cluster all"))
        self.label_channel.setText(_translate("MainWindow", "Load data first"))
        self.pushButton_gotochannel.setText(_translate("MainWindow", "Go to Channel"))
        self.pushButton_previouschannel.setText(_translate("MainWindow", "Previous"))
        self.pushButton_nextchannel.setText(_translate("MainWindow", "Next Channel"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoadFolder.setText(_translate("MainWindow", "Load folder"))
        self.actionLoadFolder.setStatusTip(_translate("MainWindow", "Load a folder"))
        self.actionLoadFolder.setShortcut(_translate("MainWindow", "Ctrl+O"))

class SW_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent = parent)
        self.setupUi(self)
        self.setup_SW()

    def assist_openFileNameDialog(self):
        dlg = QFileDialog()
        if dlg.exec_():
           fileName = dlg.selectedFiles()
           if fileName:
               self.filenow = fileName[0]
               self.assist_loadfile() # import the first file
    def assist_loadfile(self):
        self.is_loaddata = True
        filename = self.filenow
        self.statusbar.showMessage(f"loading file: {filename}")
        td = sio.loadmat(filename)
        self.rawmat = td
        self.data = td.get('waveforms')
        self.assist_computedefault()

    def assist_computedefault(self):
        # compute PCA
        waves = self.data['waves'].item()
        self.pca = self.PCA(waves)
        self.sw_combobox_pc()
        self.sw_allfigures()
    def sw_load_folder(self):
        """
            temporary: load a single file
        """
        self.assist_openFileNameDialog()

    """
    Incomplete
    """
    def sw_plt_units(self):
        waves = self.data['waves'].item()
        units = self.data['units'].item()
        for ui in range(self.n_maxunit):
            if ui == self.unit_now:
                self.units_axes[0, ui].getViewBox().setBackgroundColor("m")
            else:
                self.units_axes[0, ui].getViewBox().setBackgroundColor("w")
            if any(units == ui):
                tid = (units == ui).squeeze()
                lines = MultiLine(waves[tid,])
                lines.setcolor(self.color_unit[ui])
                self.units_axes[0, ui].clear()
                self.units_axes[0, ui].addItem(lines)
            else:
                self.units_axes[0, ui].clear()
    """
    Siyu codes
    """

    def keyPressEvent(self, event):
        key = event.key()
        str = chr(key)
        if str.isdigit():
            keyint = np.int64(str)
            if (keyint >=1) & (keyint <=self.n_maxunit):
                self.unit_now = keyint
                self.sw_allfigures()

    def sw_plt_selected(self):
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
    def update_unit(self, units):
        self.data['units'].itemset(units)
        self.sw_allfigures()
        self.sw_autosave()

    def assist_addpointsinpolygon(self, pts):
        poly_path = mplPath.Path(pts)
        pc = self.pc_now
        idp = poly_path.contains_points(pc)
        if len(self.idx_selected) > 0:
            self.idx_selected = idp
        else:
            self.idx_selected = idp
        self.sw_allfigures()
    def assist_addpointsinline(self, pts):
        waves = self.data['waves'].item()
        nl = waves.shape[0]
        npixel = waves.shape[1]
        idp = np.repeat(False, nl)
        for i in range(nl):
            te = intersection(pts[:, 0], pts[:, 1], list(range(npixel)),waves[i,])
            idp[i] = (te[0].shape[0] > 0)
        if len(self.idx_selected) > 0:
            self.idx_selected = idp
        else:
            self.idx_selected = idp
        self.sw_allfigures()

    def mouse_clicked_pca(self, event):
        if self.is_addpoint != 0:
            p = self.graphicsView_pca.plotItem.vb.mapSceneToView(event.scenePos())
            self.pca_polygon_vertices.append([p.x(), p.y()])
            pts = np.reshape(self.pca_polygon_vertices, [-1, 2])
            pts = np.append(pts, [pts[0,]], axis=0)
            self.assist_addpointsinpolygon(pts)
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
    def sw_addpoint(self):
        cursor = QtCore.Qt.CrossCursor # QCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)
        self.assist_reset_polygon()
        self.is_addpoint = 1
    def sw_removepoint(self):
        cursor = QtCore.Qt.CrossCursor # QCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)
        self.assist_reset_polygon()
        self.is_addpoint = -1
    def sw_plt_raw(self):
        idp = self.idx_selected
        waves = self.data['waves'].item()
        units = self.data['units'].item()
        self.graphicsView_raw.clear()
        for ui in range(self.n_maxunit):
            tid = (units == ui).squeeze()
            if (self.is_addpoint != 0) & (len(idp) > 0):
                tid = tid & ~idp
            if any(tid):
                lines = MultiLine(waves[tid, ])
                lines.setcolor(self.color_unit[ui])
                self.graphicsView_raw.addItem(lines)
        if (self.is_addpoint != 0) & (len(idp) > 0) & any(idp):
            lines = MultiLine(waves[idp,])
            lines.setcolor("m")
            self.graphicsView_raw.addItem(lines)
    def sw_plt_pca(self):
        pc = self.pca
        n1 = self.comboBox_PC1.currentText()
        n1 = int(n1[2])-1
        n2 = self.comboBox_PC2.currentText()
        n2 = int(n2[2])-1
        units = self.data['units'].item()
        for ui in range(self.n_maxunit):
            if any(units == ui):
                tid = (units == ui).squeeze()
                self.pca_scatter[ui].setData(x = pc[tid,n1], y = pc[tid,n2])
            else:
                self.pca_scatter[ui].setData(x=[], y=[])
        pc = self.pc_now
        idp = self.idx_selected
        if len(idp) > 0:
            self.points_selected.setData(x=pc[idp, 0], y=pc[idp, 1])
        else:
            self.points_selected.setData(x=[], y=[])
    def sw_allfigures(self):
        self.sw_plt_pca()
        self.sw_plt_raw()
        self.sw_plt_selected()
        self.sw_plt_units()
    def sw_confirm(self):
        idp = self.idx_selected
        units = self.data['units'].item()
        if (len(idp) > 0) & any(idp):
            if self.is_addpoint == 1:
                units[idp] = self.unit_now
            else:
                if self.is_addpoint == -1:
                    units[idp] = 0
        self.is_addpoint = 0
        self.idx_selected = []
        self.update_unit(units)
        self.assist_reset_polygon()
        cursor = QtCore.Qt.QCursor
        self.graphicsView_pca.setCursor(cursor)
        self.graphicsView_raw.setCursor(cursor)

    def sw_reset(self):
        self.setup_reset()
        units = self.data['units'].item()
        units = np.zeros(units.shape)
        units = np.int64(units)
        self.update_unit(units)
    def sw_autosave(self):
        mdict = self.rawmat
        mdict['waveforms'] = self.data
        sio.savemat(self.filenow, mdict)
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
    def setup_axes(self):
        # set up graphics view background
        self.graphicsView_pca.setBackground('w')
        self.graphicsView_raw.setBackground('w')
        self.graphicsView_units.setBackground('w')
        # set up lines
        # -- raw
        self.raw_emptyplot = self.graphicsView_raw.plot(x=[], y=[], pen=pg.mkPen("m"))
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

    def setup_connect(self):
        self.actionLoadFolder.triggered.connect(self.sw_load_folder)
        self.pushButton_reset.clicked.connect(self.sw_reset)
        self.pushButton_Add.clicked.connect(self.sw_addpoint)
        self.pushButton_Remove.clicked.connect(self.sw_removepoint)
        self.pushButton_Confirm.clicked.connect(self.sw_confirm)
        self.pca_emptyplot.scene().sigMouseClicked.connect(self.mouse_clicked_pca)
        self.raw_emptyplot.scene().sigMouseClicked.connect(self.mouse_clicked_raw)
        self.comboBox_PC1.activated.connect(self.sw_combobox_pc)
        self.comboBox_PC2.activated.connect(self.sw_combobox_pc)

    def sw_combobox_pc(self):
        if self.is_loaddata:
            n1 = self.comboBox_PC1.currentText()
            n1 = int(n1[2]) - 1
            n2 = self.comboBox_PC2.currentText()
            n2 = int(n2[2]) - 1
            pc = self.pca[:, (n1, n2)]
            self.pc_now = pc
            self.assist_reset_polygon()
            self.sw_plt_pca()
    def assist_reset_polygon(self):
        self.pca_polygon_vertices = []
        self.raw_line_vertices = []
    def setup_reset(self):
        self.assist_reset_polygon()
        self.comboBox_PC1.setCurrentIndex(0)
        self.comboBox_PC2.setCurrentIndex(1)
        self.sw_combobox_pc()
        self.is_addpoint = 0
        self.idx_selected = []
        self.unit_now = 0
    def setup_SW(self):
        # set up colors
        self.color_unit = ["k","r","b","g","c","y"]
        self.n_maxunit = len(self.color_unit)
        self.is_loaddata = False
        self.setup_reset()
        self.setup_axes()
        self.setup_connect()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = SW_MainWindow()
    ui.show()
    ui.filenow = './test.mat'
    ui.assist_loadfile()
    sys.exit(app.exec_())
