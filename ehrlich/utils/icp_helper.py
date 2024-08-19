import sys
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

import numpy as np
import time


CORRESPONDENCE_THRESHOLD = 15


def get_min_distance_correspondence(dists, threshold=CORRESPONDENCE_THRESHOLD):

    """For each point in P find the closest one in Q."""

    corresp = np.argmin(dists, axis=1)
    indices = np.arange(len(corresp))
    mask = dists[np.arange(dists.shape[0]), corresp] < threshold
    filtered_indices = indices[mask]
    filtered_corresp = corresp[mask]
    res_corresp = np.column_stack((filtered_indices, filtered_corresp))

    # print(f"f{res_corresp=}")
    # print(f"f{len(res_corresp)=}")

    return res_corresp


def center_data(data, exclude_indices=[]):
    """
    Find geometry center and center all data
    :param data: data to center
    :param exclude_indices: indices to exclude
    :return: center and center of data
    """
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=1)]).T
    return center, data - center


def compute_cross_covariance(P, Q, correspondences):
    p = P.T[[correspondences[:, 0]], :][0]
    q = Q.T[[correspondences[:, 1]], :][0]

    mean_p = np.mean(p, axis=0)
    mean_q = np.mean(q, axis=0)

    p_centered = p - mean_p
    q_centered = q - mean_q

    covariance_matrix = np.dot(p_centered.T, q_centered)

    return covariance_matrix


def icp_svd(P, Q, iterations=10, kernel=lambda diff: 1.0):
    """ Perform ICP using SVD. """
    # center_of_Q, Q_centered = center_data(Q)
    norm_value = 0.
    P_copy = P.copy()
    corresp_values = None
    exclude_indices = []
    dists = []

    for i in range(iterations):
        print(f"icp iteration {i}")
        # center_of_P, P_centered = center_data(P_copy, exclude_indices=exclude_indices)

        # delta = P_copy.T.reshape(-1, 1, 3) - Q_centered.T
        delta = P_copy.T.reshape(-1, 1, 3) - Q.T
        dists = np.linalg.norm(delta, axis=-1)

        correspondences = get_min_distance_correspondence(dists)
        corresp_values = correspondences

        P_indices = correspondences[:, 0]
        Q_indices = correspondences[:, 1]
        norm_value = np.mean(dists[P_indices, Q_indices])

        # cov, exclude_indices = compute_cross_covariance(P_copy, Q_centered, correspondences, kernel)
        cov = compute_cross_covariance(P_copy, Q, correspondences)
        U, S, V_T = np.linalg.svd(cov)
        R = U.dot(V_T)
        # t = center_of_Q - R.dot(center_of_P)
        # P_copy = R.dot(P_copy) + t
        P_copy = R.dot(P_copy)

    corresp_values2 = get_min_distance_correspondence(dists.T)

    return P_copy, norm_value, corresp_values, corresp_values2


def icp_optimization(coords_list1: np.ndarray, coords_list2: np.ndarray, iterations: int) -> (np.ndarray, float, List[Tuple[int, int]]):
    """
    Compute best icp alignment
    """

    p = coords_list1.T
    q = coords_list2.T

    p_values, norm_values, corresp_values, corresp_values2 = icp_svd(p, q, iterations=iterations)
    out_coords = p_values.T

    return out_coords, norm_values, corresp_values, corresp_values2



