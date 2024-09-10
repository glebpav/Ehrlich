import time

import numpy as np

from ehrlich.operations import find_closest_points
from ehrlich.utils.displacement import Rotation, Displacement

CORRESPONDENCE_THRESHOLD = 15


def get_min_distance_correspondence(dists, threshold=CORRESPONDENCE_THRESHOLD):

    """For each point in P find the closest one in Q."""

    corresp = np.argmin(dists, axis=1)
    indices = np.arange(len(corresp))
    mask = dists[np.arange(dists.shape[0]), corresp] < threshold
    filtered_indices = indices[mask]
    filtered_corresp = corresp[mask]
    res_corresp = np.column_stack((filtered_indices, filtered_corresp))
    return res_corresp


def compute_cross_covariance(P, Q, correspondences):
    p = P.T[[correspondences[:, 0]], :][0]
    q = Q.T[[correspondences[:, 1]], :][0]

    mean_p = np.mean(p, axis=0)
    mean_q = np.mean(q, axis=0)

    p_centered = p - mean_p
    q_centered = q - mean_q

    covariance_matrix = np.dot(p_centered.T, q_centered)

    return covariance_matrix


def icp_step(p_coords: np.ndarray, q_coords: np.ndarray) -> (np.ndarray, float, np.ndarray, np.ndarray, Displacement):

    correspondences, dists = find_closest_points(p_coords.astype(np.float32), q_coords.astype(np.float32))
    correspondences2, _ = find_closest_points(q_coords, p_coords)

    delta = p_coords.T.reshape(-1, 1, 3) - q_coords.T
    dists = np.linalg.norm(delta, axis=-1)

    cov = compute_cross_covariance(p_coords, q_coords, correspondences)
    U, S, V_T = np.linalg.svd(cov)
    R = U.dot(V_T)
    p_coords = R.dot(p_coords)
    norm_value = np.mean(dists)
    displacement = Rotation(R, second_place=True)

    return p_coords, norm_value, correspondences, correspondences2, displacement


