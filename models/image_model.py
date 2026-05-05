import cv2

from core.global_thresholding import apply_global_threshold
from core.local_thresholding import apply_local_threshold
from core.optimal_thresholding import compute_optimal_threshold


class ImageModel:
    def __init__(self):
        self.original_image = None  # Stores the cv2 numpy array
        self.processed_image = None

    def threshold_image(self, mode, technique, window_size=3):
        if self.original_image is None:
            raise ValueError("No image loaded")

        gray_image = self._to_grayscale(self.original_image)
        threshold_function = self._get_threshold_function(technique)

        if mode == "Global":
            return apply_global_threshold(gray_image, threshold_function)
        elif mode == "Local":
            block_size = max(3, window_size)
            return apply_local_threshold(gray_image, threshold_function, block_size=block_size)
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")

    def segment_image(self, method="KMeans", params=None):
        """
        Segments self.original_image using the selected method.

        Args:
            method (str): One of "KMeans", "MeanShift", "Agglomerative", "RegionGrowing"
            params (dict): Method-specific parameters.

        Returns:
            Segmented image as a BGR numpy uint8 array.
        """
        if self.original_image is None:
            raise ValueError("No image loaded")

        if params is None:
            params = {}

        method = method.strip()

        if method == "KMeans":
            from core.kmeans_segmentation import kmeans_segment
            k = params.get("k", 5)
            return kmeans_segment(self.original_image, k=k)

        elif method == "MeanShift":
            from core.meanshift_segmentation import meanshift_segment
            spatial_radius = params.get("spatial_radius", 7)
            color_radius   = params.get("color_radius", 6.5)
            min_region     = params.get("min_region", 20)
            return meanshift_segment(
                self.original_image,
                spatial_radius=spatial_radius,
                color_radius=color_radius,
                min_region=min_region,
            )

        elif method == "Agglomerative":
            from core.agglomerative_segmentation import agglomerative_segment
            n_clusters    = params.get("n_clusters", 4)
            linkage       = params.get("linkage", "ward")
            resize_dim_val = params.get("resize_dim", 80)
            return agglomerative_segment(
                self.original_image,
                n_clusters=n_clusters,
                linkage=linkage,
                resize_dim=(resize_dim_val, resize_dim_val),
            )

        elif method == "RegionGrowing":
            from core.region_growing_segmentation import region_growing_segment
            n_seeds   = params.get("n_seeds", 5)
            threshold = params.get("threshold", 15.0)
            return region_growing_segment(
                self.original_image,
                n_seeds=n_seeds,
                threshold=threshold,
            )

        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    # =============== PRIVATE HELPERS ===============

    def _to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _get_threshold_function(self, technique):
        technique = technique.lower()

        if technique == "otsu":
            return self._otsu_threshold
        if technique == "optimal":
            return self._optimal_threshold
        if technique == "spectral":
            return self._spectral_threshold

        raise ValueError(f"Unknown threshold technique: {technique}")

    def _otsu_threshold(self, image):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

    def _optimal_threshold(self, image):
        return compute_optimal_threshold(image)

    def _spectral_threshold(self, image):
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return threshold