from core.kmeans_segmentation import kmeans_segment
from core.meanshift_segmentation import meanshift_segment
from core.agglomerative_segmentation import agglomerative_segment

_SEG_METHOD_PAGE = {
    "KMeans": 0,
    "MeanShift": 1,
    "Agglomerative": 2,
}

class SegmentationController:
    def __init__(self, ui, model, statusbar=None):
        self.ui = ui
        self.model = model
        self.statusbar = statusbar
        self._connect_signals()
        # Set initial stack page
        self._on_method_changed(self.ui.comboSegMethod.currentText())

    def _connect_signals(self):
        self.ui.comboSegMethod.currentTextChanged.connect(self._on_method_changed)

    def _on_method_changed(self, method):
        page = _SEG_METHOD_PAGE.get(method, 0)
        self.ui.segParamsStack.setCurrentIndex(page)
        if self.statusbar:
            self.statusbar.showMessage(f"Segmentation method: {method}", 2000)

    def apply_segmentation(self):
        if self.model.original_image is None:
            raise ValueError("No image loaded")

        method = self.ui.comboSegMethod.currentText()
        params = self._collect_params(method)

        if self.statusbar:
            self.statusbar.showMessage(f"Running {method} segmentation…", 0)

        return self.model.segment_image(method=method, params=params)

    def _collect_params(self, method):
        if method == "KMeans":
            return {"k": self.ui.spinKValue.value()}
        elif method == "MeanShift":
            return {
                "spatial_radius": self.ui.spinSpatialRadius.value(),
                "color_radius":   self.ui.spinColorRadius.value(),
                "min_region":     self.ui.spinMinRegion.value(),
            }
        elif method == "Agglomerative":
            return {
                "n_clusters": self.ui.spinNClusters.value(),
                "linkage":    self.ui.comboLinkage.currentText(),
                "resize_dim": self.ui.spinResizeDim.value(),
            }
        return {}