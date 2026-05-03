from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFile, QMetaObject
from PySide6.QtGui import QAction
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QMenu, QMenuBar, QStatusBar,
    QWidget, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QTabWidget, QStackedWidget
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        loader = QUiLoader()
        ui_path = Path(__file__).with_name("main_window.ui")

        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.OpenModeFlag.ReadOnly):
            raise FileNotFoundError(ui_path)

        self.centralwidget = loader.load(ui_file, MainWindow)
        ui_file.close()

        if self.centralwidget is None:
            raise RuntimeError("Failed to load UI")

        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.resize(1366, 768)

        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # ================= WIDGETS FROM UI =================
        # Top bar buttons
        self.btnUploadOriginal = self.centralwidget.findChild(QPushButton, "btnUploadOriginal")
        self.btnReset = self.centralwidget.findChild(QPushButton, "btnReset")

        # Image labels
        self.lblOriginal = self.centralwidget.findChild(QWidget, "lblOriginal")
        self.lblProcessed = self.centralwidget.findChild(QWidget, "lblProcessed")

        # Sidebar
        self.parametersStack = self.centralwidget.findChild(QStackedWidget, "parametersStack")
        self.tabs = self.centralwidget.findChild(QTabWidget, "tabs")

        # ---- Threshold tab controls ----
        self.comboThresholdMode = self.centralwidget.findChild(QComboBox, "comboThresholdMode")
        self.comboThresholdTechnique = self.centralwidget.findChild(QComboBox, "comboThresholdTechnique")
        self.windowSize = self.centralwidget.findChild(QSpinBox, "windowSize")
        self.btnApplyThreshold = self.centralwidget.findChild(QPushButton, "btnApplyThreshold")

        # ---- Segmentation tab controls ----
        self.btnApplySegmentation = self.centralwidget.findChild(QPushButton, "btnApplySegmentation")

        # Method selector + per-method parameter stack
        self.comboSegMethod = self.centralwidget.findChild(QComboBox, "comboSegMethod")
        self.segParamsStack = self.centralwidget.findChild(QStackedWidget, "segParamsStack")

        # KMeans params
        self.spinKValue = self.centralwidget.findChild(QSpinBox, "spinKValue")

        # MeanShift params
        self.spinSpatialRadius = self.centralwidget.findChild(QSpinBox, "spinSpatialRadius")
        self.spinColorRadius = self.centralwidget.findChild(QDoubleSpinBox, "spinColorRadius")
        self.spinMinRegion = self.centralwidget.findChild(QSpinBox, "spinMinRegion")

        # Agglomerative params
        self.spinNClusters = self.centralwidget.findChild(QSpinBox, "spinNClusters")
        self.comboLinkage = self.centralwidget.findChild(QComboBox, "comboLinkage")
        self.spinResizeDim = self.centralwidget.findChild(QSpinBox, "spinResizeDim")

        QMetaObject.connectSlotsByName(MainWindow)