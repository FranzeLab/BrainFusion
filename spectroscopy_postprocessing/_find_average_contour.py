import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import find_contours
from scipy.interpolate import interp1d


def extract_and_interpolate_contours(masks, num_points=1000):
    """Extract and interpolate contours from binary masks."""
    contours = []
    for mask in masks:
        contour = extract_contours(mask)
        contour[-1, :] = contour[0, :]  # Close contour
        contour_interp = interpolate_contour(contour, num_points)
        contours.append(contour_interp)
    return contours


def extract_contours(mask):
    """Extract contours from a binary mask."""
    contours = find_contours(mask, 0.5)

    if len(contours) > 1:
        contour = np.vstack(contours)
    else:
        contour = contours[0]

    # Switch the x and y indices (swap columns)
    contour[:, [0, 1]] = contour[:, [1, 0]]

    return contour


def interpolate_contour(contour, num_points):
    """Interpolate a contour to have a fixed number of points."""
    dists = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    interp_x = interp1d(cumulative_dists, contour[:, 1], kind='linear')
    interp_y = interp1d(cumulative_dists, contour[:, 0], kind='linear')
    new_dists = np.linspace(0, cumulative_dists[-1], num_points)
    return np.vstack((interp_y(new_dists), interp_x(new_dists))).T


def match_contours(contours):
    """Match contours to the template using ellipse fitting."""
    template_contour = contours[0]
    matched_contours = [template_contour]  # Keep template as the first contour
    for contour in contours[1:]:
        shifted_contour = shift_contour(contour, template_contour)  # ToDO: Sometimes makes trafo matrix singular
        matched_contour = match_contour_with_ellipse(contour, template_contour)
        matched_contours.append(matched_contour)
    return matched_contours


def shift_contour(contour, reference_contour):
    """Shift contour so that its starting point is closest to the reference contour."""
    # Calculate the Euclidean distance between the starting point of reference_contour and all points in contour
    distances = np.linalg.norm(contour - reference_contour[0], axis=1)

    # Find the index of the point closest to the starting point of the reference_contour
    shift_index = np.argmin(distances)

    # Circularly shift the contour to start from that point
    shifted_contour = np.roll(contour, -shift_index, axis=0)

    return shifted_contour


def match_contour_with_ellipse(A, B):
    """Match contour A to contour B using ellipse fitting and rotation."""
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
    A_rotated = rotate_contour(A, np.deg2rad(rotation_angle), center_A)

    return A_rotated + (center_B - center_A)


def fit_ellipse(contour):
    """Fit an ellipse to the given contour and return its parameters."""
    if len(contour) < 5:
        return None
    return cv2.fitEllipse(contour.astype(np.float32))


def rotate_contour(contour, angle, center):
    """Rotate the contour around a center point by a specified angle."""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return (contour - center).dot(R.T) + center


def calculate_median_contour(contours):
    """Calculate the median contour from a list of contours."""
    contour = np.median(np.array(contours), axis=0)
    contour[-1, :] = contour[0, :]  # Close contour
    return contour


def plot_contours(template_contour, matched_contours, median_contour):
    """Plot the template, matched contours, and the median contour"""
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot the template contour
    plt.plot(template_contour[:, 0], template_contour[:, 1], 'g-', linewidth=2, label='Mask (Template)')

    # Plot the matched contours (only label the first one)
    for i, contour in enumerate(matched_contours[1:]):
        if i == 0:
            plt.plot(contour[:, 0], contour[:, 1], 'b-', label='Masks (Matched)')
        else:
            plt.plot(contour[:, 0], contour[:, 1], 'b-')  # No label for subsequent contours

    # Plot the median contour
    plt.plot(median_contour[:, 0], median_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    plt.legend()
    plt.axis('equal')
    plt.title('Path Density with Contours')

    return fig


def find_average_contour(masks):
    # Extract and interpolate contours
    contours_list = extract_and_interpolate_contours(masks)

    # Circularly align and match contours using ellipse fitting
    matched_contour_list = match_contours(contours_list)

    # Calculate the median contour
    median_contour = calculate_median_contour(matched_contour_list)

    return median_contour, contours_list
