import numpy as np
import cv2


class UnionFind:
    """Helper Structure for Transitive Closure Region Merging"""
    def __init__(self, n):
        self.parent = np.arange(n)
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            if root_i < root_j: self.parent[root_j] = root_i
            else: self.parent[root_i] = root_j


class MeanShift:
    def __init__(self, spatial_radius=7, color_radius=6.5, min_region=20, max_iters=100):
        """
        Initializes the Mean Shift Segmenter with algorithmic parameters.
        """
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.color_rad_sq = color_radius ** 2
        self.min_region = min_region
        self.max_iters = max_iters

    def _filter(self, image):
        """
        Phase 1: Mean Shift Filter
        Filters the image using a circular flat kernel and color distance in L*u*v colorspace.
        """
        print("1/3 Running Mean Shift Filter...")
        img_f32 = image.astype(np.float32) / 255.0
        luv_img = cv2.cvtColor(img_f32, cv2.COLOR_RGB2Luv)

        h, w = luv_img.shape[:2]
        filtered_luv = np.zeros_like(luv_img)

        Y, X = np.indices((h, w))

        for j in range(h):
            for i in range(w):
                ic, jc = i, j
                L, U, V = luv_img[j, i]

                shift = 5.0
                iters = 0

                while shift > 1.0 and iters < self.max_iters:
                    rmin, rmax = max(0, jc - self.spatial_radius), min(h, jc + self.spatial_radius + 1)
                    cmin, cmax = max(0, ic - self.spatial_radius), min(w, ic + self.spatial_radius + 1)

                    window_luv = luv_img[rmin:rmax, cmin:cmax]
                    window_x = X[rmin:rmax, cmin:cmax]
                    window_y = Y[rmin:rmax, cmin:cmax]

                    dL = window_luv[:, :, 0] - L
                    dU = window_luv[:, :, 1] - U
                    dV = window_luv[:, :, 2] - V
                    dist_sq = dL**2 + dU**2 + dV**2

                    mask = dist_sq <= self.color_rad_sq
                    num_pixels = np.sum(mask)

                    if num_pixels == 0:
                        break

                    mi = np.sum(window_x[mask]) / num_pixels
                    mj = np.sum(window_y[mask]) / num_pixels
                    mL = np.sum(window_luv[:, :, 0][mask]) / num_pixels
                    mU = np.sum(window_luv[:, :, 1][mask]) / num_pixels
                    mV = np.sum(window_luv[:, :, 2][mask]) / num_pixels

                    ic_new = int(mi + 0.5)
                    jc_new = int(mj + 0.5)
                    di, dj = ic_new - ic, jc_new - jc
                    dl_diff, du_diff, dv_diff = mL - L, mU - U, mV - V

                    shift = (di**2 + dj**2 + dl_diff**2 + du_diff**2 + dv_diff**2)

                    ic, jc = ic_new, jc_new
                    L, U, V = mL, mU, mV
                    iters += 1

                filtered_luv[j, i] = [L, U, V]

        return filtered_luv

    def _cluster(self, luv_image):
        """
        Phase 2: Mean Shift Cluster
        Clusters the L*u*v image using 8-connected neighbors and region growing.
        """
        print("2/3 Clustering Image...")
        h, w = luv_image.shape[:2]
        labels = np.full((h, w), -1, dtype=np.int32)
        lbl = -1

        modes = []
        mode_points = []
        dxdy = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for j in range(h):
            for i in range(w):
                if labels[j, i] < 0:
                    lbl += 1
                    labels[j, i] = lbl

                    L, U, V = luv_image[j, i]
                    mode_sum = np.array([L, U, V], dtype=np.float64)
                    pts_count = 1

                    stack = [(i, j)]

                    while stack:
                        cx, cy = stack.pop()
                        for dx, dy in dxdy:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                if labels[ny, nx] < 0:
                                    nL, nU, nV = luv_image[ny, nx]
                                    dist2 = (nL - L)**2 + (nU - U)**2 + (nV - V)**2

                                    if dist2 < self.color_rad_sq:
                                        labels[ny, nx] = lbl
                                        stack.append((nx, ny))
                                        mode_sum += np.array([nL, nU, nV])
                                        pts_count += 1

                    modes.append(mode_sum / pts_count)
                    mode_points.append(pts_count)

        return labels, np.array(modes), np.array(mode_points)

    def _build_adjacency(self, labels):
        """Builds Region Adjacency List for borders between regions"""
        adj = set()
        v_edges = np.c_[labels[:-1, :].ravel(), labels[1:, :].ravel()]
        adj.update(map(tuple, np.sort(v_edges[v_edges[:, 0] != v_edges[:, 1]], axis=1)))
        h_edges = np.c_[labels[:, :-1].ravel(), labels[:, 1:].ravel()]
        adj.update(map(tuple, np.sort(h_edges[h_edges[:, 0] != h_edges[:, 1]], axis=1)))
        return adj

    def _transitive_closure(self, labels, modes, mode_points):
        """
        Phase 3: Transitive Closure
        Merges similar adjacent regions and prunes smaller, spurious regions.
        """
        print("3/3 Applying Transitive Closure & Pruning...")
        region_count = len(modes)

        # 1. Transitive Closure (Merge similar adjacent regions up to 5 times)
        for _ in range(5):
            adj = self._build_adjacency(labels)
            uf = UnionFind(region_count)

            for r1, r2 in adj:
                dist2 = np.sum((modes[r1] - modes[r2])**2)
                if dist2 < self.color_rad_sq:
                    uf.union(r1, r2)

            new_labels_map = np.array([uf.find(i) for i in range(region_count)])
            unique_labels = np.unique(new_labels_map)

            if len(unique_labels) == region_count:
                break

            new_modes = np.zeros((len(unique_labels), 3))
            new_mode_points = np.zeros(len(unique_labels), dtype=np.int32)
            label_mapping = {root: new_id for new_id, root in enumerate(unique_labels)}

            for i in range(region_count):
                new_id = label_mapping[new_labels_map[i]]
                new_mode_points[new_id] += mode_points[i]
                new_modes[new_id] += modes[i] * mode_points[i]

            for i in range(len(unique_labels)):
                new_modes[i] /= new_mode_points[i]

            labels = np.vectorize(lambda x: label_mapping[new_labels_map[x]])(labels)
            modes, mode_points = new_modes, new_mode_points
            region_count = len(modes)

        # 2. Prune small regions
        while True:
            small_regions = np.where(mode_points < self.min_region)[0]
            if len(small_regions) == 0:
                break

            adj = self._build_adjacency(labels)
            adj_list = {i: [] for i in range(region_count)}
            for r1, r2 in adj:
                adj_list[r1].append(r2)
                adj_list[r2].append(r1)

            uf = UnionFind(region_count)
            merged = False

            for sr in small_regions:
                if uf.find(sr) != sr: continue
                neighbors = adj_list[sr]
                if not neighbors: continue

                best_n = neighbors[np.argmin([np.sum((modes[sr] - modes[n])**2) for n in neighbors])]
                uf.union(sr, best_n)
                merged = True

            if not merged:
                break

            new_labels_map = np.array([uf.find(i) for i in range(region_count)])
            unique_labels = np.unique(new_labels_map)

            new_modes = np.zeros((len(unique_labels), 3))
            new_mode_points = np.zeros(len(unique_labels), dtype=np.int32)
            label_mapping = {root: new_id for new_id, root in enumerate(unique_labels)}

            for i in range(region_count):
                new_id = label_mapping[new_labels_map[i]]
                new_mode_points[new_id] += mode_points[i]
                new_modes[new_id] += modes[i] * mode_points[i]

            for i in range(len(unique_labels)):
                if new_mode_points[i] > 0:
                    new_modes[i] /= new_mode_points[i]

            labels = np.vectorize(lambda x: label_mapping[new_labels_map[x]])(labels)
            modes, mode_points = new_modes, new_mode_points
            region_count = len(modes)

        return labels, modes

    def process(self, image):
        """
        Runs the full Mean Shift segmentation pipeline on the given RGB image.
        Returns the Filtered image and Segmented image (both in RGB format).
        """
        # 1. Filter Phase
        filtered_luv = self._filter(image)

        # 2. Cluster Phase
        labels, modes, mode_points = self._cluster(filtered_luv)

        # 3. Transitive Closure & Prune Phase
        final_labels, final_modes = self._transitive_closure(labels, modes, mode_points)

        # 4. Reconstruct Image from modes
        segmented_luv = np.zeros_like(filtered_luv)
        for i in range(len(final_modes)):
            segmented_luv[final_labels == i] = final_modes[i]

        # 5. Convert Luv results back to standard RGB 8-bit mapping
        filtered_rgb = cv2.cvtColor(filtered_luv, cv2.COLOR_Luv2RGB)
        filtered_rgb = np.clip(filtered_rgb * 255, 0, 255).astype(np.uint8)

        segmented_rgb = cv2.cvtColor(segmented_luv, cv2.COLOR_Luv2RGB)
        segmented_rgb = np.clip(segmented_rgb * 255, 0, 255).astype(np.uint8)

        return filtered_rgb, segmented_rgb


def meanshift_segment(image_bgr, spatial_radius=7, color_radius=6.5, min_region=20):
    """
    Segments a BGR image using the custom MeanShift implementation.
    Returns a BGR segmented image (numpy uint8 array).
    """
    scale_factor = 0.5
    resized_img = cv2.resize(
        image_bgr, 
        dsize=None, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv2.INTER_AREA
    )
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    ms = MeanShift(spatial_radius=spatial_radius, color_radius=color_radius, min_region=min_region)
    _, segmented_rgb = ms.process(img_rgb)
    return cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)