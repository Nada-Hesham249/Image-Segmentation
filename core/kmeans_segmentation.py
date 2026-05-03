import numpy as np


class KMeansCustom:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        # dict of indices [0-K-1]: for each cluster idx: list of its points
        self.clusters = {}
        # List of indices [0-K-1]: for each cluster idx: D-dimensional vector for centroid
        self.centroids = []

    def initialize_centroids(self, data):
        # select k random different centroids: assumes k <= # number of examples
        random_indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[random_indices]

    def _dist(self, point, centroid):
        #return np.linalg.norm(point - centroid)    # has sqrt
        return np.sum((point - centroid)**2)        # we don't need actual distance

    def assign_clusters(self, data):
        self.clusters = {i: [] for i in range(self.k)}
        for point in data:
            distances = [self._dist(point, centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            self.clusters[cluster_idx].append(point)    # add cluster points

    def update_centroids(self):
        for cluster_idx, cluster_points in self.clusters.items():
            # Average points of each cluster. axis=0 ==> vertically
            self.centroids[cluster_idx] = np.mean(cluster_points, axis=0)

    def fit(self, data):
        self.initialize_centroids(data)

        for _ in range(self.max_iter):
            self.assign_clusters(data)
            prev_centroids = np.copy(self.centroids)
            self.update_centroids()

            # Check for convergence
            if np.allclose(self.centroids, prev_centroids, rtol=1e-4):
                break

    def predict(self, data):
        predictions = []
        for point in data:
            distances = [self._dist(point, centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            predictions.append(cluster_idx)
        return predictions


def kmeans_segment(image_bgr, k=5, num_pixels=10000):
    """
    Segments a BGR image using the custom KMeans implementation.
    Returns a BGR segmented image (numpy uint8 array).
    """
    import cv2
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_int = img_rgb.astype('int')

    np.random.seed(17)
    h, w, c = img_int.shape
    data = img_int.reshape(-1, c)

    # Pick only random pixels to train the model
    num_pixels = min(num_pixels, data.shape[0])
    random_indices = np.random.choice(data.shape[0], size=num_pixels, replace=False)
    sample_data = data[random_indices]

    # Fit on the tiny sample
    kmeans = KMeansCustom(k=k)
    kmeans.fit(sample_data)

    # Fast Assignment using NumPy broadcasting
    centroids = np.array(kmeans.centroids)
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    predicted_labels = np.argmin(distances, axis=1)

    # Reconstruct the image
    segmented_data = centroids[predicted_labels]
    segmented_img = segmented_data.reshape(h, w, c)

    if segmented_img.max() > 1.0:
        segmented_img = np.clip(segmented_img, 0, 255).astype(np.uint8)
    else:
        segmented_img = np.clip(segmented_img, 0.0, 1.0)

    # Convert back to BGR for Qt display
    return cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)