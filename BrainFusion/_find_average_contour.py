import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist
from shapely.geometry import Polygon


def extract_and_interpolate_contours(masks, num_points):
    contours = []
    for mask in masks:
        contour = extract_contours(mask)
        contour_interp = interpolate_contour(contour, num_points)
        contours.append(contour_interp)
    return contours


def extract_contours(mask):
    # Ensure mask is in the correct format (uint8)
    mask = (mask > 0).astype(np.uint8)

    # Find the external contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming there is only one main contour, return the largest contour
        contour = max(contours, key=cv2.contourArea)
        # Remove the unnecessary dimension
        contour = np.squeeze(contour)

        # Close contour
        contour[-1, :] = contour[0, :]  # Close contour
    else:
        contour = None

    return contour


def interpolate_contour(contour, num_points):
    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_y(new_dists), interp_x(new_dists))).T


def match_contours(contours, template_contour, fit_routine='ellipse'):
    matched_contours = []
    for i, contour in enumerate(contours):
        if fit_routine == 'ellipse':
            matched_contour, _, _, _ = match_contour_with_ellipse(contour, template_contour)
        else:
            matched_contour = contour
            print("No fit routine applied to match contours!")

        matched_contours.append(matched_contour)
    return matched_contours


def shift_contour(contour, reference_contour):
    # Calculate the Euclidean distance between the starting point of reference_contour and all points in contour
    distances = np.linalg.norm(contour - reference_contour[0], axis=1)

    # Find the index of the point closest to the starting point of the reference_contour
    shift_index = np.argmin(distances)

    # Circularly shift the contour to start from that point
    shifted_contour = np.roll(contour, -shift_index, axis=0)

    return shifted_contour


def match_contour_with_ellipse(A, B):
    ellipse_A = fit_ellipse(A)
    ellipse_B = fit_ellipse(B)
    if ellipse_A is None or ellipse_B is None:
        print("No ellipse could be fitted to the contours! Continue with original shapes.")
        return A  # Return original if ellipse can't be fitted
    center_A = np.array(ellipse_A[0])
    center_B = np.array(ellipse_B[0])
    rotation_angle = ellipse_B[2] - ellipse_A[2]  # Angle difference
    if 90 < rotation_angle:
        rotation_angle = rotation_angle - 180
    elif rotation_angle < -90:
        rotation_angle = rotation_angle + 180
    rotation_angle = np.deg2rad(rotation_angle)
    A_rotated = rotate_coordinate_system(A, rotation_angle, center_A)

    return A_rotated + (center_B - center_A), rotation_angle, center_A, center_B


def fit_ellipse(contour):
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour.astype(np.float32))


def rotate_coordinate_system(contour, angle, center):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return (contour - center).dot(R.T) + center


def get_contour_orientation(contour):
    # Calculate the signed area using the Shoelace formula
    x, y = contour[:, 0], contour[:, 1]
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    assert signed_area != 0, print("Signed area enclosed by contour is exactly zero!")
    return 'clockwise' if signed_area < 0 else 'counterclockwise'


def orientate_shift_contours(contours, template_index=0):
    assert 0 <= template_index < len(contours), print("Contour template index out of range!")
    template_contour = contours[template_index]
    modified_contours = []
    orientation = get_contour_orientation(template_contour)
    for i, contour in enumerate(contours):
        if i == template_index:
            modified_contours.append(template_contour)
            continue

        # Shift contour
        shifted_contour = shift_contour(contour, template_contour)

        # Orient contour
        if get_contour_orientation(contour) != orientation:
            shifted_contour = shifted_contour[::-1]  # Reverse the order

        modified_contours.append(shifted_contour)

    return modified_contours, template_contour


def calculate_average_contour(contours, average='median'):
    if average == 'median':
        contour = np.median(np.array(contours), axis=0)
    elif average == 'mean':
        contour = np.mean(np.array(contours), axis=0)
    else:
        raise Exception(f'{average} is not defined for contour averaging.')
    contour[-1, :] = contour[0, :]  # Close contour
    return contour


def calculate_distances(contours, master_contour, metric='jaccard'):
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

    def frechet_distance(curve_a, curve_b):
        try:
            frechet_dist = frdist(np.array(curve_a), np.array(curve_b))
        except RuntimeWarning:
            print('Recursion error while calculating Frechet_distance!')
            frechet_dist = np.nan

        return frechet_dist

    def hausdorff_distance(curve_a, curve_b):
        return max(
            directed_hausdorff(curve_a, curve_b)[0],
            directed_hausdorff(curve_b, curve_a)[0]
        )

    results = []
    for i, contour in enumerate(contours):
        if metric == 'jaccard':
            dist = jaccard_distance(contour, master_contour)

        elif metric == 'frechet':
            # Fréchet distance
            dist = frechet_distance(contour, master_contour)

        elif metric == 'hausdorff':
            # Hausdorff distance
            dist = hausdorff_distance(contour, master_contour)

        else:
            raise Exception(f'{metric} is not defined as an error metric.')

        # Append distances for the current curve
        results.append(dist)

    return results


def find_average_contour(contours_list, template_index=0, fit_routine='ellipse', average='mean',
                         metric='jaccard'):
    for element in contours_list:
        assert isinstance(element, np.ndarray), "Each contour must be a numpy array!"
        assert element.ndim == 2 and element.shape[1] == 2, "Each contour must be an Nx2 array!"

    # Orient and circularly align contours to match template
    contours_list, template_contour = orientate_shift_contours(contours_list, template_index)

    #  Match contours using ellipse fitting
    matched_contour_list = match_contours(contours_list, template_contour, fit_routine)

    # Calculate the median contour
    avg_contour = calculate_average_contour(matched_contour_list, average=average)

    # Calculate error
    errors = calculate_distances(matched_contour_list, avg_contour, metric=metric)

    return avg_contour, contours_list, template_contour, matched_contour_list, errors
