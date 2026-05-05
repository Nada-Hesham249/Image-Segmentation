import numpy as np
import cv2
from collections import deque


class RegionGrower:
    """
    Multi-seed region growing segmentation.

    Each seed spawns its own region. Pixels are added to a region when
    their intensity (or color) difference from the region's running mean
    is within `threshold`.  Unvisited pixels after all seeds are exhausted
    are left unlabelled (label = -1) and painted with their original color.
    """

    def __init__(self, n_seeds: int = 5, threshold: float = 15.0):
        """
        Args:
            n_seeds   : number of seed points (chosen by uniform grid sampling)
            threshold : max Euclidean distance in color space from region mean
                        for a neighbour to be accepted into the region
        """
        self.n_seeds = n_seeds
        self.threshold = threshold

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: H×W×3 BGR uint8 numpy array

        Returns:
            Segmented H×W×3 BGR uint8 numpy array where every pixel in a
            region is replaced by that region's mean color.
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        seeds = self._pick_seeds(h, w)
        labels = np.full((h, w), -1, dtype=np.int32)
        region_colors = self._grow(img_rgb, seeds, labels)

        return self._reconstruct(image_bgr, img_rgb, labels, region_colors)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _pick_seeds(self, h: int, w: int):
        """
        Place seeds on a uniform grid so they spread across the image.
        Falls back to random if n_seeds is large relative to image size.
        """
        # grid approach: divide image into n_seeds roughly equal tiles
        cols = max(1, int(np.ceil(np.sqrt(self.n_seeds * w / h))))
        rows = max(1, int(np.ceil(self.n_seeds / cols)))

        ys = np.linspace(0, h - 1, rows + 2, dtype=int)[1:-1]
        xs = np.linspace(0, w - 1, cols + 2, dtype=int)[1:-1]

        seeds = [(int(y), int(x)) for y in ys for x in xs]

        # trim or pad to exactly n_seeds
        if len(seeds) > self.n_seeds:
            seeds = seeds[: self.n_seeds]
        elif len(seeds) < self.n_seeds:
            # add random extras (avoid duplicates)
            existing = set(seeds)
            rng = np.random.default_rng(42)
            while len(seeds) < self.n_seeds:
                y = int(rng.integers(0, h))
                x = int(rng.integers(0, w))
                if (y, x) not in existing:
                    seeds.append((y, x))
                    existing.add((y, x))

        return seeds

    def _grow(self, img_rgb: np.ndarray, seeds, labels: np.ndarray):
        """
        BFS region growing from every seed simultaneously (round-robin
        so no single seed monopolises the image).

        Returns:
            region_colors: list of mean RGB arrays, one per seed/region
        """
        h, w = img_rgb.shape[:2]
        n = len(seeds)

        # running sums for mean-colour tracking
        color_sum = [np.zeros(3, dtype=np.float64) for _ in range(n)]
        pixel_count = [0] * n
        region_colors = [np.zeros(3, dtype=np.float64) for _ in range(n)]

        queues = [deque() for _ in range(n)]
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # initialise each seed
        for idx, (sy, sx) in enumerate(seeds):
            labels[sy, sx] = idx
            c = img_rgb[sy, sx].astype(np.float64)
            color_sum[idx] += c
            pixel_count[idx] = 1
            region_colors[idx] = c
            queues[idx].append((sy, sx))

        active = list(range(n))

        while active:
            still_active = []
            for idx in active:
                if not queues[idx]:
                    continue
                cy, cx = queues[idx].popleft()
                mean_c = color_sum[idx] / pixel_count[idx]

                for dy, dx in neighbours:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == -1:
                        nc = img_rgb[ny, nx].astype(np.float64)
                        if np.linalg.norm(nc - mean_c) <= self.threshold:
                            labels[ny, nx] = idx
                            color_sum[idx] += nc
                            pixel_count[idx] += 1
                            region_colors[idx] = color_sum[idx] / pixel_count[idx]
                            queues[idx].append((ny, nx))

                if queues[idx]:
                    still_active.append(idx)

            active = still_active

        return region_colors

    def _reconstruct(self, image_bgr, img_rgb, labels, region_colors):
        """Paint each labelled pixel with its region's mean color."""
        h, w = img_rgb.shape[:2]
        out_rgb = img_rgb.copy().astype(np.float64)

        for idx, mean_c in enumerate(region_colors):
            mask = labels == idx
            out_rgb[mask] = mean_c

        # unlabelled pixels keep original color (already set from img_rgb copy)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# module-level convenience function (matches style of other segmenters)
# --------------------------------------------------------------------------
def region_growing_segment(image_bgr: np.ndarray,
                           n_seeds: int = 5,
                           threshold: float = 15.0) -> np.ndarray:
    """
    Segments a BGR image using multi-seed region growing.

    Args:
        image_bgr : H×W×3 BGR uint8 numpy array
        n_seeds   : number of seed points
        threshold : colour similarity threshold for region acceptance

    Returns:
        Segmented BGR uint8 numpy array
    """
    grower = RegionGrower(n_seeds=n_seeds, threshold=threshold)
    return grower.segment(image_bgr)