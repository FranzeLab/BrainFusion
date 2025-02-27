import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist
from shapely.geometry import Polygon, Point, LineString
from skimage.transform import AffineTransform
from scipy.stats import linregress
from collections import defaultdict


def interpolate_contour(contour, num_points):
    assert isinstance(contour, np.ndarray), "Each contour must be a numpy array!"
    assert contour.ndim == 2 and contour.shape[1] == 2, "Each contour must be an Nx2 array!"

    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_y(new_dists), interp_x(new_dists))).T


def align_contours(contour_list, grid_list, template_index, fit_routine='ellipse'):
    # Circularly shift contours to match template and transform all contours to their geometric centre
    modified_contours, centres = circularly_shift_contours(contour_list, template_index)

    # Apply translations to grids
    modified_grids = [grid - centre for grid, centre in zip(grid_list, centres)]

    #  Match contours using fitting routine
    matched_contours, matched_grids = match_contours(modified_contours,
                                                     modified_grids,
                                                     modified_contours[template_index],
                                                     fit_routine)

    return matched_contours, matched_grids


def align_sc_contours(contour_list, grid_list, init_points=None, template_index=0):
    # Circularly shift contours to match template and transform all contours to their geometric centre
    shifted_contours, _ = circularly_shift_contours(contour_list,
                                                    template_index=template_index,
                                                    centre_contours=False,
                                                    init_points=init_points)

    #  Match contours using bbox fitting routine and DTW alignment
    matched_contours = []
    dtw_contours = []
    matched_grids = []
    for i, contour in enumerate(shifted_contours):
        contour_trafo, template_trafo = match_contour_with_bbox(contour, shifted_contours[template_index])
        matched_contour = contour_trafo(contour)
        matched_grid = contour_trafo(grid_list[i])
        matched_contours.append(matched_contour)
        matched_grids.append(matched_grid)

    for i, contour in enumerate(matched_contours):
        contour_dtw, _ = dtw_with_tangent_penalty(contour, matched_contours[template_index])
        dtw_contours.append(contour_dtw)

    return dtw_contours, matched_grids


def circularly_shift_contours(contours, template_index=0, centre_contours=True, init_points=None):
    assert 0 <= template_index < len(contours), print("Contour template index out of range!")
    assert not (init_points is not None and centre_contours is True), (
        'Attention: Centering contours while using predefined'
        ' shift coordinates!')

    template_contour = contours[template_index]
    template_centre = np.mean(template_contour, axis=0)
    if centre_contours:
        template_contour = template_contour - template_centre
    modified_contours = []
    centres = []

    # Get topological orientation of template contour
    orientation = get_contour_orientation(template_contour)

    for i, contour in enumerate(contours):
        #if i == template_index:
        #    modified_contours.append(template_contour)
        #    centres.append(template_centre)
        #    continue

        # Centre contours around geometric mean
        contour_centre = np.mean(contour, axis=0)
        if centre_contours:
            contour = contour - contour_centre

        if init_points is not None:
            # Find the closest initial point in init_points
            distances_to_init_points = np.linalg.norm(contour - init_points[i], axis=1)
            shift_index = np.argmin(distances_to_init_points)  # Find the closest point
        else:
            # Calculate the Euclidean distance between the starting point of template_contour and all points in contour
            distances = np.linalg.norm(contour - template_contour[0], axis=1)

            # Find the index of the point closest to the starting point of template_contour
            shift_index = np.argmin(distances)

        # Circularly shift the contour to start from the selected point
        shifted_contour = np.roll(contour, -shift_index, axis=0)

        # Topologically orient contour
        if get_contour_orientation(contour) != orientation:
            shifted_contour = shifted_contour[::-1]  # Reverse the order

        # Close contours
        shifted_contour[-1, :] = shifted_contour[0, :]

        modified_contours.append(shifted_contour)
        centres.append(contour_centre)

    return modified_contours, centres


def get_contour_orientation(contour):
    # Calculate the signed area using the Shoelace formula
    x, y = contour[:, 0], contour[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    assert signed_area != 0, print("Signed area enclosed by contour is exactly zero!")
    return 'clockwise' if signed_area < 0 else 'counterclockwise'


def match_contours(contours, grids, template_contour, fit_routine='ellipse'):
    matched_contours = []
    matched_grids = []
    for i, contour in enumerate(contours):
        if fit_routine == 'ellipse':
            matched_contour, matched_grid, _, _, _ = match_contour_with_ellipse(contour, template_contour, grids[i])
        else:
            matched_contour = contour
            matched_grid = grids[i]
            print("No fit routine applied to match contours!")

        matched_contours.append(matched_contour)
        matched_grids.append(matched_grid)
    return matched_contours, matched_grids


def match_contour_with_ellipse(A, B, grid):
    ellipse_A = fit_ellipse(A)
    ellipse_B = fit_ellipse(B)
    if ellipse_A is None or ellipse_B is None:
        print("No ellipse could be fitted to the contours! Continue with original shapes.")
        return A, None, None, None  # Return original if ellipse can't be fitted
    center_A = np.array(ellipse_A[0])  # Important: Ellipse fit centre does not always align with geometric mean!
    center_B = np.array(ellipse_B[0])
    rotation_angle = ellipse_B[2] - ellipse_A[2]  # Angle difference
    if rotation_angle > 90:
        rotation_angle = rotation_angle - 180
    elif rotation_angle < -90:
        rotation_angle = rotation_angle + 180
    rotation_angle = np.deg2rad(rotation_angle)
    A_rotated = rotate_coordinate_system(A, rotation_angle, center_A)
    grid_rotated = rotate_coordinate_system(grid, rotation_angle, center_A)

    return A_rotated, grid_rotated, rotation_angle, center_A, center_B


def fit_ellipse(contour):
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour.astype(np.float32))


def rotate_coordinate_system(contour, angle, center):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return (contour - center).dot(R.T) + center


def match_contour_with_bbox(a, b):
    # Translate contours to their geometric centre
    centre_a = -np.mean(a, axis=0)
    a_cent = a + centre_a
    centre_b = -np.mean(b, axis=0)
    b_cent = b + centre_b

    # Get boundary box corner points
    source_points = extract_bbox_corners(a_cent)
    target_points = extract_bbox_corners(b_cent)

    # Scale in x and y
    dist_x_target = (target_points[3] - target_points[0])[0]
    dist_x_source = (source_points[3] - source_points[0])[0]
    scale_x = dist_x_target / dist_x_source if dist_x_source != 0 else 1

    dist_y_target = (target_points[1] - target_points[0])[1]
    dist_y_source = (source_points[1] - source_points[0])[1]
    scale_y = dist_y_target / dist_y_source if dist_y_source != 0 else 1

    rot_ang = regression_alignment_angle(a_cent, b_cent)

    # Translate contours to match their first coordinates
    translate_a = [0, 0]  # b_cent[0, :] - a_cent[0, :]

    # Define Transformation matrix for a and b
    affine_transformation_a = (AffineTransform(translation=(centre_a[0], centre_a[1])) +
                               AffineTransform(scale=(scale_x, scale_y)) +
                               AffineTransform(rotation=rot_ang) +
                               AffineTransform(translation=(translate_a[0], translate_a[1]))
                               )
    affine_transformation_b = AffineTransform(translation=(centre_b[0], centre_b[1]))

    return affine_transformation_a, affine_transformation_b


def extract_bbox_corners(contour):
    """Extract key points from the four corners of the bounding box."""
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    return np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])


def regression_alignment_angle(source_contour, target_contour):
    """Fit a line to the right side of the contours and compute rotation."""
    right_source = source_contour[source_contour[:, 0] > np.percentile(source_contour[:, 0], 98)]
    right_target = target_contour[target_contour[:, 0] > np.percentile(target_contour[:, 0], 98)]
    slope_source, _, _, _, _ = linregress(right_source[:, 0], right_source[:, 1])
    slope_target, _, _, _, _ = linregress(right_target[:, 0], right_target[:, 1])
    angle = np.arctan(slope_target) - np.arctan(slope_source)
    # Adjust angle to wrap around within the range of ±π/2
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi

    return 0  #angle


def dtw_with_tangent_penalty(contour1, contour2, base_lambda_tangent=0.4):
    """Match contours using DTW with a dynamically normalized tangent similarity constraint."""
    n, m = len(contour1), len(contour2)

    # Compute local tangent vectors
    tangents1 = np.diff(contour1, axis=0, append=contour1[:1])
    tangents2 = np.diff(contour2, axis=0, append=contour2[:1])

    # Normalize lambda_tangent dynamically
    spatial_dists = [np.linalg.norm(contour1[i] - contour2[j]) for i in range(n) for j in range(m)]
    tangent_diffs = [1 - np.dot(tangents1[i], tangents2[j]) / (np.linalg.norm(tangents1[i]) * np.linalg.norm(tangents2[j]) + 1e-10)
                     for i in range(n) for j in range(m)]

    median_spatial = np.median(spatial_dists)
    median_tangent = np.median(tangent_diffs)

    lambda_tangent = base_lambda_tangent * (median_spatial / (median_tangent + 1e-10))

    def cost_function(i, j):
        """Cost function incorporating spatial distance and normalized tangent similarity."""
        p1, p2 = contour1[i], contour2[j]
        t1, t2 = tangents1[i], tangents2[j]

        spatial_dist = np.linalg.norm(p1 - p2)
        tangent_penalty = lambda_tangent * (1 - np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-10))
        return spatial_dist + tangent_penalty

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


def find_average_contour(contours_list, average='star_domain', metric='jaccard'):
    # Calculate the average contour
    avg_contour = calculate_average_contour(contours_list, average=average)

    # Calculate error
    errors = calculate_distances(contours_list, avg_contour, metric=metric)

    return avg_contour, errors


def calculate_average_contour(contours, average='star_domain', num_bins=360):
    if average == 'star_domain':
        def is_star_domain(contour, centre, tol=10):
            polygon = Polygon(contour)  # Convert contour to a polygon
            center_point = Point(centre)

            # Step 1: Check if the center is inside the polygon
            if not polygon.contains(center_point):
                return False

            # Step 2: Check if all boundary points are visible from the center
            for point in contour:
                boundary_point = Point(point)
                line_of_sight = LineString([centre, point])

                # Compute intersections with the polygon boundary
                intersection = polygon.boundary.intersection(line_of_sight)

                # Convert intersection points to a list of coordinates
                if isinstance(intersection, Point):  # Single intersection
                    intersections = [intersection]
                elif hasattr(intersection, "geoms"):  # Multiple intersections
                    intersections = list(intersection.geoms)
                else:
                    raise Exception("An error occurred while checking if polygon is star domain.")

                distances = [boundary_point.distance(inter) for inter in intersections]
                if not all(d < tol for d in distances):
                    return False

            return True

        # Check if all contours are star domains with respect to the given centre
        assert all(is_star_domain(contour, [0, 0]) for contour in contours), ('Not all contours define star '
                                                                              'domains with respect to their geometric '
                                                                              'centre!')

        # Convert contours to polar coordinates
        angles = []
        radii = []
        for contour in contours:
            x, y = contour[:, 0], contour[:, 1]
            theta = np.arctan2(contour[:, 1], contour[:, 0])  # Compute angles
            r = np.sqrt(x ** 2 + y ** 2)
            angles.append(theta)
            radii.append(r)

        # Create a uniform grid of angles
        angle_bins = np.linspace(-np.pi, np.pi, num_bins)

        # Interpolate radii for each contour onto the uniform grid
        interpolated_radii = np.array([
            np.interp(angle_bins, np.sort(theta), r[np.argsort(theta)])
            for theta, r in zip(angles, radii)
        ])

        # Compute the mean radius at each angle bin
        avg_radii = np.mean(interpolated_radii, axis=0)

        # Convert mean polar coordinates back to Cartesian
        avg_x = avg_radii * np.cos(angle_bins)
        avg_y = avg_radii * np.sin(angle_bins)

        contour = np.column_stack((avg_x, avg_y))

    elif average == 'median':
        contour = np.median(np.array(contours), axis=0)

    elif average == 'mean':
        contour = np.mean(np.array(contours), axis=0)

    else:
        raise Exception(f'{average} is not defined for contour averaging.')

    # Close contour
    contour[-1, :] = contour[0, :]
    return contour


def calculate_distances(contours, master_contour, metric='jaccard'):
    results = []
    for i, contour in enumerate(contours):
        if metric == 'jaccard':
            def jaccard_distance(curve_a, curve_b):
                # Create Polygon objects
                polya = Polygon(curve_a)
                polyb = Polygon(curve_b)

                # Calculate the intersection and union of the polygons
                intersection_area = polya.intersection(polyb).area
                union_area = polya.union(polyb).area

                # Calculate the Jaccard Index based on areas
                jaccard_index = intersection_area / union_area

                # Calculate the Jaccard Distance
                jaccard_distance = 1 - jaccard_index
                return jaccard_distance

            dist = jaccard_distance(contour, master_contour)

        elif metric == 'frechet':
            def frechet_distance(curve_a, curve_b):
                try:
                    frechet_dist = frdist(np.array(curve_a), np.array(curve_b))
                except RuntimeWarning:
                    print('Recursion error while calculating Frechet_distance!')
                    frechet_dist = np.nan
                return frechet_dist

            # Fréchet distance
            dist = frechet_distance(contour, master_contour)

        elif metric == 'hausdorff':
            def hausdorff_distance(curve_a, curve_b):
                return max(
                    directed_hausdorff(curve_a, curve_b)[0],
                    directed_hausdorff(curve_b, curve_a)[0]
                )

            # Hausdorff distance
            dist = hausdorff_distance(contour, master_contour)

        else:
            raise Exception(f'{metric} is not defined as an error metric.')

        # Append distances for the current curve
        results.append(dist)

    return results
