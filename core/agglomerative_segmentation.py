import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeImageSegmenter:
    """
    A class to perform Image Segmentation using Agglomerative Clustering.
    """

    def __init__(self, n_clusters=3, linkage='ward', resize_dim=(100, 100)):
        """
        Initializes the segmenter.

        Args:
            n_clusters (int): The number of clusters to form.
            linkage (str): The linkage criterion ('ward', 'average', 'complete', 'single').
                           - 'ward' minimizes WCSS (inter-cluster distance).
                           - 'average' uses the average distance.
                           - 'complete' uses the farthest distance.
                           - 'single' uses the nearest distance.
            resize_dim (tuple): Dimensions to resize the image to prevent MemoryErrors.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.resize_dim = resize_dim

        # Initialize the scikit-learn model
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)

    def segment(self, image_bgr):
        """
        Accepts a BGR numpy array, performs agglomerative clustering,
        and returns the segmented image as a BGR numpy uint8 array.
        """
        # Convert from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_shape = image.shape

        # 2. Resize image to handle computational complexity
        if self.resize_dim:
            processed_image = cv2.resize(image, self.resize_dim)
        else:
            processed_image = image

        # 3. Flatten the image into a 2D array of pixels (N_pixels, 3_color_channels)
        pixels = processed_image.reshape((-1, 3))

        # 4. Fit the model and assign each pixel to a cluster
        # This is the bottom-up approach merging closest clusters
        labels = self.model.fit_predict(pixels)

        # 5. Reconstruct the image
        # Replace each pixel's color with the average color of its assigned cluster
        segmented_pixels = np.zeros_like(pixels)
        for cluster_idx in range(self.n_clusters):
            # Find all pixels belonging to the current cluster
            cluster_mask = (labels == cluster_idx)

            if np.any(cluster_mask):
                # Calculate the mean color for this cluster
                mean_color = pixels[cluster_mask].mean(axis=0)
                segmented_pixels[cluster_mask] = mean_color

        # 6. Reshape back to the 2D image dimensions
        segmented_image = segmented_pixels.reshape(processed_image.shape)

        # 7. Resize back to original dimensions using nearest neighbor to preserve sharp cluster edges
        if self.resize_dim:
            segmented_image = cv2.resize(segmented_image,
                                         (original_shape[1], original_shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

        # Convert back to BGR for Qt display
        return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)


def agglomerative_segment(image_bgr, n_clusters=6, linkage='ward', resize_dim=(100, 100)):
    """
    Segments a BGR image using AgglomerativeImageSegmenter.
    Returns a BGR segmented image (numpy uint8 array).
    """
    segmenter = AgglomerativeImageSegmenter(
        n_clusters=n_clusters,
        linkage=linkage,
        resize_dim=resize_dim
    )
    return segmenter.segment(image_bgr)