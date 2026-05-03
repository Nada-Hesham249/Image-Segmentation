import cv2
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from utils.converters import cv_to_pixmap
from controllers.threshold_controller import ThresholdController
from controllers.segmentation_controller import SegmentationController


class MainController:
    def __init__(self, ui, model, window):
        self.ui = ui
        self.model = model
        self.window = window
        self.threshold_controller = ThresholdController(ui, model, self.ui.statusbar)
        self.segmentation_controller = SegmentationController(ui, model, self.ui.statusbar)
        self._connect_signals()

    # ================= CONNECT =================
    def _connect_signals(self):

        # self.ui.actionOpen_Image.triggered.connect(self.load_image)
        # self.ui.actionExit.triggered.connect(self.window.close)

        self.ui.btnUploadOriginal.clicked.connect(self.load_image)
        self.ui.btnReset.clicked.connect(self.reset_view)

        # 🆕 new UI buttons
        self.ui.btnApplyThreshold.clicked.connect(self.apply_thresholding)
        self.ui.btnApplySegmentation.clicked.connect(self.apply_segmentation)

    # ================= LOAD IMAGE =================
    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        self.model.original_image = cv2.imread(file_path)

        self.display_original_image(self.model.original_image)
        self.ui.statusbar.showMessage(f"Loaded: {file_path}", 3000)

    # ================= RESET =================
    def reset_view(self):

        self.ui.lblOriginal.clear()
        self.ui.lblOriginal.setText("No Image")

        self.ui.lblProcessed.clear()
        self.ui.lblProcessed.setText("No Image")

        self.model.original_image = None
        self.model.processed_image = None

        self.ui.statusbar.showMessage("Reset done", 2000)

    # ================= THRESHOLDING =================
    def apply_thresholding(self):

        if self.model.original_image is None:
            self.ui.statusbar.showMessage("Load image first", 3000)
            return

        self.model.processed_image = self.threshold_controller.apply_thresholding()
        self.display_processed_image(self.model.processed_image)

    # ================= SEGMENTATION =================
    def apply_segmentation(self):
        if self.model.original_image is None:
            self.ui.statusbar.showMessage("Load image first", 3000)
            return
        try:
            self.model.processed_image = self.segmentation_controller.apply_segmentation()
            self.display_processed_image(self.model.processed_image)
            self.ui.statusbar.showMessage("Segmentation applied", 3000)
        except Exception as e:
            self.ui.statusbar.showMessage(f"Segmentation error: {e}", 5000)

    # ================= DISPLAY =================
    def display_original_image(self, img):

        pix = cv_to_pixmap(img)
        if pix:
            self.ui.lblOriginal.setPixmap(
                pix.scaled(
                    self.ui.lblOriginal.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def display_processed_image(self, img):

        pix = cv_to_pixmap(img)
        if pix:
            self.ui.lblProcessed.setPixmap(
                pix.scaled(
                    self.ui.lblProcessed.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )