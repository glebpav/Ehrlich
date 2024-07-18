import sys
from abc import ABC
from typing import Union, Tuple, List

import numpy as np
from matplotlib import pyplot as plt

# from ehrlich.segment import Segment
from ehrlich.utils.amin_similarity import get_amin_idx, amin_similarity_matrix
from ehrlich.utils.icp_helper import icp_optimization
from ehrlich.utils.math_utils import get_rotated_vector

CLOSENESS_THRESHOLD = 1.  # in Angstrom


class Alignment(ABC):
    def __init__(self, segment1: "Segment", segment2: "Segment"):
        self.segment1: "Segment" = segment1
        self.segment2: "Segment" = segment2
        self.segment1_new_coords: Union[np.ndarray, None] = None
        self.segment2_new_coords: Union[np.ndarray, None] = None
        self.correspondence: Union[List[Tuple[int, int]], None] = None
        self.amin_sim: Union[float, None] = None
        self.norm_dist: Union[float, None] = None

    def z_axis_alignment(
            self,
            aligning_points: np.ndarray,
            origin_point_coords: np.ndarray,
            origin_norm: np.ndarray
    ) -> np.ndarray:

        """
        Compute moved and rotated coords to make origin point in (0; 0; 0) and its norm (0; 0; 1)
        :return: np.ndarray of computed coords
        """

        e3 = origin_norm
        g = np.array([10, 0, -(e3[0] / e3[2])])
        e1 = g / np.linalg.norm(g)
        e2 = np.cross(e3, e1)

        t = np.array([e1, e2, e3]).T
        t_inv = np.linalg.inv(t)

        out_coords = []
        for point in aligning_points:
            out_coords.append(np.matmul(t_inv, point - origin_point_coords))

        return np.array(out_coords)

    def icp_alignment(
            self,
            coords1: np.ndarray,
            coords2: np.ndarray,
            rotation_list: List[int]
    ):

        min_norm_value = sys.float_info.max
        out_corresp = None
        best_rotated_coords = None  # coords of segment1 in best rotation
        out_coords = None  # coords of segment2 in best icp alignment for `best_rotated_coords`

        for rotation_idx, rotation_angle in enumerate(rotation_list):
            print(f"rotation_idx: {rotation_idx}")
            rotated_coords = np.array([
                get_rotated_vector(vector, rotation_angle, "z")
                for vector in coords1
            ])
            coords, norm_values, corresp_values = icp_optimization(coords2, rotated_coords)
            if min_norm_value > norm_values:
                min_norm_value = norm_values
                out_coords = coords
                best_rotated_coords = rotated_coords
                out_corresp = corresp_values

        amin_score = 0.
        for idx1, idx2 in out_corresp:
            acid_idx1 = get_amin_idx(self.segment1.mol.resnames[self.segment1.mol.vamap[idx1]])
            acid_idx2 = get_amin_idx(self.segment1.mol.resnames[self.segment1.mol.vamap[idx2]])
            amin_score += amin_similarity_matrix[acid_idx1][acid_idx2]

        self.segment2_new_coords = out_coords
        self.segment1_new_coords = best_rotated_coords
        self.amin_sim = amin_score / len(out_corresp)
        self.correspondence = out_coords
        self.norm_dist = min_norm_value


class SegmentAlignment(Alignment):
    """
    Detailed icp alignment for 2 segments
    """
    def __init__(self, segment1: "Segment", segment2: "Segment", rotation_list):
        """
        :param segment1: first aligned segment
        :param segment2: second aligned segment
        """
        super().__init__(segment1, segment2)
        self._find_best_alignment(rotation_list)

    def _find_best_alignment(self, rotation_list: List[int]=[0, 60, 120, 180, 270]):
        """
        Align 2 segments by icp algorithm witch fills left fields in this class
        """

        aligned_coords1 = self.z_axis_alignment(
            np.array([self.segment1.mol.vcoords[point_idx] for point_idx in self.segment1.used_points]),
            self.segment1.mol.vcoords[self.segment1.origin_idx],
            self.segment1.mol.compute_norm(self.segment1.origin_idx)
        )
        aligned_coords2 = self.z_axis_alignment(
            np.array([self.segment2.mol.vcoords[point_idx] for point_idx in self.segment2.used_points]),
            self.segment2.mol.vcoords[self.segment2.origin_idx],
            self.segment2.mol.compute_norm(self.segment2.origin_idx)
        )

        self.icp_alignment(aligned_coords1, aligned_coords2, rotation_list)

    def draw(self):
        """
        Draw aligned segments using matplotlib
        """

        origin_coords1 = np.copy(self.segment1.mol.vcoords)
        origin_coords2 = np.copy(self.segment2.mol.vcoords.copy())

        for idx, point_idx in enumerate(self.segment1.used_points):
            self.segment1.mol.vcoords[point_idx] = self.segment1_new_coords[idx]
        for idx, point_idx in enumerate(self.segment2.used_points):
            self.segment2.mol.vcoords[point_idx] = self.segment2_new_coords[idx]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        self.segment1.draw(ax=ax, with_whole_surface=False)
        self.segment2.draw(ax=ax, with_whole_surface=False)
        plt.show()

        for idx, point_idx in enumerate(self.segment1.used_points):
            self.segment1.mol.vcoords[point_idx] = origin_coords1[point_idx]
        for idx, point_idx in enumerate(self.segment2.used_points):
            self.segment2.mol.vcoords[point_idx] = origin_coords2[point_idx]


class MoleculeAlignment(Alignment):
    def __init__(self, segment1: "Segment", segment2: "Segment", closeness_threshold=CLOSENESS_THRESHOLD):
        super().__init__(segment1, segment2)
        self.total_amin_sim: Union[float, None] = None
        self.total_dist: Union[float, None] = None
        self.closeness_threshold: float = closeness_threshold

        self._find_best_alignment()

    @property
    def match_area(self) -> float:

        points_dict = {point[1]: point for point in self.correspondence}
        match_area = 0

        for face_idx, points in enumerate(self.segment1.mol.faces):
            close_face = True
            for point in points:
                if np.linalg.norm(self.segment1.mol.vcoords[point] - self.segment2.mol.vcoords[points_dict[point]]) < self.closeness_threshold:
                    close_face = False
            if close_face:
                match_area += self.segment1.mol.faces_areas[face_idx]

        return match_area

    def _find_best_alignment(self):
        aligned_coords1 = self.z_axis_alignment(
            self.segment1.mol.vcoords,
            self.segment1.mol.vcoords[self.segment1.origin_idx],
            self.segment1.mol.compute_norm(self.segment1.origin_idx)
        )
        aligned_coords2 = self.z_axis_alignment(
            self.segment2.mol.vcoords,
            self.segment2.mol.vcoords[self.segment2.origin_idx],
            self.segment2.mol.compute_norm(self.segment2.origin_idx)
        )

        self.icp_alignment(aligned_coords1, aligned_coords2, [0, 120, 240])
        self.total_amin_sim = len(self.correspondence) * self.amin_sim







