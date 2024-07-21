import sys
from typing import List, Tuple

import numpy as np


def get_correspondence_indices(P, Q):
    """For each point in P find the closest one in Q."""

    delta = P.T.reshape(-1, 1, 3) - Q.T
    dists = np.linalg.norm(delta, axis=-1)
    corresp = np.argmin(dists, axis=1)
    indices = np.arange(len(corresp))
    res_corresp = np.column_stack((indices, corresp))

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


def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
    cov = np.zeros((3, 3))
    exclude_indices = []
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        weight = kernel(p_point - q_point)
        if weight < 0.01: exclude_indices.append(i)
        cov += weight * q_point.dot(p_point.T)
    return cov, exclude_indices


def icp_svd(P, Q, iterations=10, kernel=lambda diff: 1.0):
    """ Perform ICP using SVD. """
    center_of_Q, Q_centered = center_data(Q)
    norm_values = []
    P_values = [P.copy()]
    P_copy = P.copy()
    corresp_values = []
    exclude_indices = []
    for i in range(iterations):
        print(f"icp iteration {i}")
        center_of_P, P_centered = center_data(P_copy, exclude_indices=exclude_indices)
        correspondences = get_correspondence_indices(P_centered, Q_centered)
        corresp_values.append(correspondences)
        # min_len: int = min(len(P_centered[0]), len(Q_centered[0]))
        norm_values.append(np.linalg.norm(np.array([P_centered[:, i] - Q_centered[:, j] for i, j in correspondences])))
        # norm_values.append(np.linalg.norm(P_centered[:,:min_len] - Q_centered[:,:min_len]))
        cov, exclude_indices = compute_cross_covariance(P_centered, Q_centered, correspondences, kernel)
        U, S, V_T = np.linalg.svd(cov)
        R = U.dot(V_T)
        t = center_of_Q - R.dot(center_of_P)
        P_copy = R.dot(P_copy) + t
        P_values.append(P_copy)
    corresp_values.append(corresp_values[-1])
    return P_values, norm_values, corresp_values


def icp_optimization(coords_list1: np.ndarray, coords_list2: np.ndarray) -> (np.ndarray, float, List[Tuple[int, int]]):
    """
    Compute best icp alignment
    """

    p = coords_list1.T
    q = coords_list2.T

    p_values, norm_values, corresp_values = icp_svd(p, q, iterations=30)
    out_coords = np.array([p_values[-1][:, idx] for idx in range(len(coords_list1))])

    return out_coords, norm_values[-1], corresp_values[-1]



