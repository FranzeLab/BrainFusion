import numpy as np
import cv2
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
from skimage.transform import AffineTransform
from typing import List, Tuple
from collections import defaultdict
from numpy.linalg import svd


def dtw_wrapper(contour1: np.ndarray, contour2: np.ndarray,
                base_lambda_curvature: float = 0.5, alpha: float = 0.05,
                curvature_thresh: float = 1.0, min_dist_frac: float = 0.05,
                r2_threshold: float = 0.98) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform piecewise alignment of two closed 2D contours using straight segments as cut points.

    The contours are first segmented by detecting high-curvature corners and fitting straight lines between them.
    Each pair of consecutive cut points defines a compartment along the contour. For each compartment:

    - If both contours are straight between the same cut points:
        Arc-length-based alignment is performed.
    - If either contour is curved between the cut points:
        Dynamic Time Warping (DTW) with a curvature-based penalty is used.

    Parameters
    ----------
    contour1 : np.ndarray of shape (N, 2)
        First closed contour to be aligned (reference).
    contour2 : np.ndarray of shape (M, 2)
        Second closed contour to be matched against.
    base_lambda_curvature : float, optional
        Weighting factor for curvature mismatch penalty in DTW cost function.
    alpha : float, optional
        Smoothing factor for Gaussian kernel used in curvature calculation,
        proportional to average segment length.
    curvature_thresh : float, optional
        Threshold on absolute curvature for detecting corner candidates.
    min_dist_frac : float, optional
        Minimum arc-length distance between selected corners, as a fraction
        of the total contour perimeter.
    r2_threshold : float, optional
        Minimum coefficient of determination (R²) for a segment to be classified as straight.

    Returns
    -------
    warped_contour1 : np.ndarray of shape (K, 2)
        Reconstructed version of `contour1` after piecewise alignment.
    warped_contour2 : np.ndarray of shape (K, 2)
        Reconstructed and aligned version of `contour2` matched to `warped_contour1`.

    Notes
    -----
    - The contour is treated as *closed*, so compartments wrap around from the last cut point back to the first.
    - Curved compartments (where either contour is curved) are aligned using 'dtw_with_curvature_penalty'`'.
    """

    n1, n2 = len(contour1), len(contour2)
    sigma1 = alpha * (compute_contour_length(contour1) / max(n1, 1))
    sigma2 = alpha * (compute_contour_length(contour2) / max(n2, 1))
    curv1 = compute_css_curvature(contour1, sigma1)
    curv2 = compute_css_curvature(contour2, sigma2)

    corners1_mask = high_curvature_mask(contour1, curv1, curvature_thresh, min_dist_frac)
    corners2_mask = high_curvature_mask(contour2, curv2, curvature_thresh, min_dist_frac)
    _, seg_info1, _ = detect_linear_segments(contour1, corners1_mask, r2_threshold)
    _, seg_info2, _ = detect_linear_segments(contour2, corners2_mask, r2_threshold)

    cuts1 = sorted({p for s in seg_info1 if s["straight"] for p in (s["start"], s["end"])})
    cuts2 = sorted({p for s in seg_info2 if s["straight"] for p in (s["start"], s["end"])})

    #### Plotting
    import matplotlib.pyplot as plt

    def plot_contour_with_curvature(contour, curvature, corner_mask):
        """
        Plot a closed contour colour-coded by curvature, marking detected corners.

        Parameters
        ----------
        contour : np.ndarray of shape (N, 2)
            The closed contour coordinates.
        curvature : np.ndarray of shape (N,)
            Curvature values for each contour point.
        corner_mask : np.ndarray of bool of shape (N,)
            Boolean mask where True indicates a detected corner.
        """
        fig, ax = plt.subplots()
        sc = ax.scatter(contour[:, 0], contour[:, 1],
                        c=curvature, cmap='coolwarm', s=20, zorder=1)
        plt.colorbar(sc, ax=ax, label="Curvature")

        # Mark corners
        ax.scatter(contour[corner_mask, 0],
                   contour[corner_mask, 1],
                   c='lime', edgecolors='k', s=60, zorder=2, label="Corners")

        ax.set_aspect('equal')
        ax.legend()
        ax.set_title("Contour curvature with detected corners")
        plt.show()

    # Example usage:
    plot_contour_with_curvature(contour1, curv1, corners1_mask)
    ####

    num_segs = min(len(cuts1), len(cuts2))
    if num_segs < 2:
        return dtw_with_curvature_penalty(contour1, contour2,
                                          base_lambda_curvature, alpha)

    out1, out2 = [], []
    for k in range(num_segs):
        start1, end1 = cuts1[k], cuts1[(k + 1) % num_segs]
        start2, end2 = cuts2[k], cuts2[(k + 1) % num_segs]
        seg1 = contour1[seg_indices(start1, end1, n1)]
        seg2 = contour2[seg_indices(start2, end2, n2)]

        s1_straight = any(s["straight"] and s["start"] == start1 and s["end"] == end1 for s in seg_info1)
        s2_straight = any(s["straight"] and s["start"] == start2 and s["end"] == end2 for s in seg_info2)

        if s1_straight and s2_straight:
            seg_out1, seg_out2 = align_straight_segments(seg1, seg2)
        else:
            seg_out1, seg_out2 = dtw_with_curvature_penalty(seg1, seg2,
                                                            base_lambda_curvature, alpha)

        if k > 0:
            seg_out1 = seg_out1[1:]
            seg_out2 = seg_out2[1:]
        out1.append(seg_out1)
        out2.append(seg_out2)

    return np.vstack(out1), np.vstack(out2)


def dtw_with_curvature_penalty(contour1: np.ndarray, contour2: np.ndarray, base_lambda_curvature: float = 0.5,
                               alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two 2D contours using Dynamic Time Warping (DTW) with a curvature-based penalty.

    This function computes a soft point-to-point alignment between `contour1` and `contour2` using DTW.
    The cost function includes both spatial distance and a curvature mismatch penalty derived from
    Curvature Scale Space (CSS) representations.

    Parameters
    ----------
    contour1 : np.ndarray of shape (N, 2)
        First contour to be aligned (reference).
    contour2 : np.ndarray of shape (M, 2)
        Second contour to be matched against.
    base_lambda_curvature : float, optional (keyword-only)
        Weighting factor for curvature mismatch penalty in the DTW cost function.
    alpha : float, optional (keyword-only)
        Smoothing factor for Gaussian kernel used in CSS curvature calculation, proportional to arc length.

    Returns
    -------
    warped_contour1 : np.ndarray of shape (K, 2)
        Subsampled or repeated version of `contour1` that participated in the alignment.
    warped_contour2 : np.ndarray of shape (K, 2)
        Aligned and averaged version of `contour2` matched to `warped_contour1`.

    Notes
    -----
    - Curvature is computed using a Gaussian scale proportional to local arc length.
    - The resulting aligned points may contain duplicates or be averaged where `contour2` maps multiple points to the same location.
    """
    if not isinstance(contour1, np.ndarray) or contour1.ndim != 2 or contour1.shape[1] != 2:
        raise ValueError("contour1 must be a (N, 2) numpy array")
    if not isinstance(contour2, np.ndarray) or contour2.ndim != 2 or contour2.shape[1] != 2:
        raise ValueError("contour2 must be a (N, 2) numpy array")

    eps = 1e-10

    # Compute normalized sigma based on contour length
    len1, len2 = compute_contour_length(contour1), compute_contour_length(contour2)
    sigma1 = alpha * (len1 / len(contour1))
    sigma2 = alpha * (len2 / len(contour2))

    n, m = len(contour1), len(contour2)

    # Compute curvature at adaptive scales
    curvatures1 = compute_css_curvature(contour1, sigma1, eps=eps)
    curvatures2 = compute_css_curvature(contour2, sigma2, eps=eps)

    # Normalize lambda_curvature dynamically
    spatial_dists = [np.linalg.norm(contour1[i] - contour2[j]) for i in range(n) for j in range(m)]
    curvature_diffs = [np.abs(curvatures1[i] - curvatures2[j]) for i in range(n) for j in range(m)]

    median_spatial = np.median(spatial_dists)
    median_curvature = np.median(curvature_diffs)

    lambda_curvature = base_lambda_curvature * (median_spatial / (median_curvature + eps))

    def cost_function(i, j):
        """Cost function incorporating spatial distance and CSS-based curvature penalty."""
        p1, p2 = contour1[i], contour2[j]
        spatial_dist = np.linalg.norm(p1 - p2)
        curvature_penalty = lambda_curvature * np.abs(curvatures1[i] - curvatures2[j])
        return spatial_dist ** 2 + curvature_penalty

    # Initialize DP matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Compute DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_function(i - 1, j - 1)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # Insertion
                                          dtw_matrix[i, j - 1],  # Deletion
                                          dtw_matrix[i - 1, j - 1])  # Match

    # Backtrack to find the optimal path
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
        if step == 0:
            i -= 1  # Move up
        elif step == 1:
            j -= 1  # Move left
        else:
            i -= 1  # Move diagonally
            j -= 1

    path.reverse()

    idx1, idx2 = np.array(list(zip(*path)))

    # Dictionary to store values corresponding to idx1
    contour2_mapping = defaultdict(list)

    # Store values of contour2 based on alignment
    for i1, i2 in zip(idx1, idx2):
        contour2_mapping[i1].append(contour2[i2])  # Store actual contour2 values, not indices

    # Compute averaged values
    idx1_unique = np.array(list(contour2_mapping.keys()))
    contour2_averaged = np.array(
        [np.mean(values, axis=0) for values in contour2_mapping.values()])  # Averaging the points

    return contour1[idx1_unique], contour2_averaged


def compute_contour_length(contour):
    return np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1))


def gaussian_smooth_contour(contour, sigma):
    smoothed_x = ndi.gaussian_filter1d(contour[:, 0], sigma, mode='wrap')
    smoothed_y = ndi.gaussian_filter1d(contour[:, 1], sigma, mode='wrap')
    return np.column_stack((smoothed_x, smoothed_y))


def compute_css_curvature(contour, sigma, eps=1e-10):
    smoothed = gaussian_smooth_contour(contour, sigma)
    def circ_1st_gradient(c): return 0.5 * (np.roll(c, -1) - np.roll(c, 1))
    def circ_2nd_gradient(c): return np.roll(c, -1) - 2.0 * c + np.roll(c, 1)
    dx, dy = circ_1st_gradient(smoothed[:, 0]), circ_1st_gradient(smoothed[:, 1])
    ddx, ddy = circ_2nd_gradient(smoothed[:, 0]), circ_2nd_gradient(smoothed[:, 1])
    return (dx * ddy - dy * ddx) / (dx**2 + dy**2 + eps)**1.5


def high_curvature_mask(contour, curvature, curvature_thresh=1.0, min_dist_frac=0.05):
    seg_len = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    arc_len = np.concatenate(([0], np.cumsum(seg_len)))[:-1]
    perimeter = arc_len[-1] + seg_len[-1]
    abs_k = np.abs(curvature)
    candidates = np.flatnonzero(abs_k > curvature_thresh)
    if not candidates.size:
        return np.zeros(len(contour), bool)
    keep, min_arc_dist = [], min_dist_frac * perimeter
    for idx in candidates[np.argsort(abs_k[candidates])[::-1]]:
        if all(min(abs(arc_len[idx] - arc_len[k]),
                   perimeter - abs(arc_len[idx] - arc_len[k])) >= min_arc_dist
               for k in keep):
            keep.append(idx)
    mask = np.zeros(len(contour), bool)
    mask[keep] = True
    return mask


def detect_linear_segments(contour, corner_mask, r2_threshold=0.98):
    N = len(contour)
    corners = np.flatnonzero(corner_mask)
    if len(corners) < 2:
        return np.zeros(N, bool), [], np.zeros(N, int)
    seg_mask = np.zeros(N, bool)
    seg_ids = np.full(N, -1, int)
    seg_info = []
    seg_id_counter = 0
    for s, e in zip(corners, np.roll(corners, -1)):
        idx_range = np.arange(s, e+1) if s < e else np.r_[np.arange(s, N), np.arange(0, e+1)]
        pts = contour[idx_range]
        ctr = pts.mean(axis=0)
        _, _, Vt = svd(pts - ctr)
        dir_vec = Vt[0]
        proj = (pts - ctr) @ dir_vec
        recon = np.outer(proj, dir_vec) + ctr
        ss_res = np.sum((pts - recon)**2)
        ss_tot = np.sum((pts - ctr)**2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        straight = r2 >= r2_threshold
        if straight:
            seg_mask[idx_range] = True
            seg_ids[idx_range] = seg_id_counter
        seg_info.append({"start": s, "end": e, "r2": r2, "straight": straight})
        seg_id_counter += 1
    return seg_mask, seg_info, seg_ids


def seg_indices(start, end, N):
    return np.arange(start, end + 1) if start <= end else np.r_[np.arange(start, N), np.arange(0, end + 1)]


def arc_length_open(cont):
    ds = np.r_[0.0, np.linalg.norm(np.diff(cont, axis=0), axis=1)]
    return np.cumsum(ds)


def interp_along_arc(src_pts, t_targets):
    s = arc_length_open(src_pts)
    if s[-1] == 0:
        return np.repeat(src_pts[:1], len(t_targets), axis=0)
    u = s / s[-1]
    idx = np.searchsorted(u, t_targets, side='right')
    idx = np.clip(idx, 1, len(u) - 1)
    u0, u1 = u[idx - 1], u[idx]
    w = (t_targets - u0) / np.maximum(u1 - u0, 1e-12)
    p0, p1 = src_pts[idx - 1], src_pts[idx]
    return (1 - w)[:, None] * p0 + w[:, None] * p1


def align_straight_segments(seg1, seg2):
    t_uniform = np.linspace(0, 1, max(len(seg1), len(seg2)))
    return interp_along_arc(seg1, t_uniform), interp_along_arc(seg2, t_uniform)
