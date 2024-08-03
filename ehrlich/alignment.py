import math
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
    def __init__(
            self,
            segment1: "Segment",
            segment2: "Segment",
            icp_iterations: int,
            rotations_list: List[int]
    ):

        self.segment1: "Segment" = segment1
        self.segment2: "Segment" = segment2
        self.segment1_new_coords: Union[np.ndarray, None] = None
        self.segment2_new_coords: Union[np.ndarray, None] = None
        self.correspondence: Union[List[Tuple[int, int]], None] = None
        self.amin_sim: Union[float, None] = None
        self.geom_dist: Union[float, None] = None
        self.icp_iterations = icp_iterations

        if rotations_list is None:
            print("none case")
            rotations_list = [0]

        self.rotations_list = rotations_list

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
    ):

        min_norm_value = sys.float_info.max
        out_corresp = None
        best_rotated_coords = None  # coords of segment1 in best rotation
        out_coords = None  # coords of segment2 in best icp alignment for `best_rotated_coords`

        for rotation_idx, rotation_angle in enumerate(self.rotations_list):
            print(f"rotation_idx: {rotation_idx}")

            cos = math.cos(math.radians(rotation_angle))
            sin = math.sin(math.radians(rotation_angle))

            rotated_coords = coords1 @ np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

            coords, norm_values, corresp_values = icp_optimization(coords2, rotated_coords, self.icp_iterations)
            if min_norm_value > norm_values:
                min_norm_value = norm_values
                out_coords = coords
                best_rotated_coords = rotated_coords
                out_corresp = corresp_values

        self.segment2_new_coords = out_coords
        self.segment1_new_coords = best_rotated_coords
        self.correspondence = out_corresp
        self.geom_dist = min_norm_value

    def compute_amin_sim(self):

        amin_score = 0.
        for idx1, idx2 in self.correspondence:
            acid_idx1 = get_amin_idx(self.segment1.mol.resnames[self.segment1.mol.vamap[idx1]])
            acid_idx2 = get_amin_idx(self.segment2.mol.resnames[self.segment2.mol.vamap[idx2]])
            amin_score += amin_similarity_matrix[acid_idx1][acid_idx2]

        self.amin_sim = amin_score / len(self.correspondence)

    def _draw(self, with_whole_surface: bool, alpha: float):
        """
        Draw aligned segments using matplotlib
        """

        origin_coords1 = np.copy(self.segment1.mol.vcoords)
        origin_coords2 = np.copy(self.segment2.mol.vcoords)

        if not with_whole_surface:

            for idx, point_idx in enumerate(self.segment1.used_points):
                self.segment1.mol.vcoords[point_idx] = self.segment1_new_coords[idx]
            for idx, point_idx in enumerate(self.segment2.used_points):
                self.segment2.mol.vcoords[point_idx] = self.segment2_new_coords[idx]

        elif (len(self.segment1.mol.vcoords) == len(self.segment1_new_coords)
              and len(self.segment2.mol.vcoords) == len(self.segment2_new_coords)):

            self.segment1.mol.vcoords = self.segment1_new_coords
            self.segment2.mol.vcoords = self.segment2_new_coords

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        self.segment1.draw(ax=ax, with_whole_surface=with_whole_surface, segment_alpha=alpha, color="red")
        self.segment2.draw(ax=ax, with_whole_surface=with_whole_surface, segment_alpha=alpha, color="green")
        plt.show()

        if not with_whole_surface:
            for idx, point_idx in enumerate(self.segment1.used_points):
                self.segment1.mol.vcoords[point_idx] = origin_coords1[point_idx]
            for idx, point_idx in enumerate(self.segment2.used_points):
                self.segment2.mol.vcoords[point_idx] = origin_coords2[point_idx]

        elif (len(self.segment1.mol.vcoords) == len(self.segment1_new_coords)
              and len(self.segment2.mol.vcoords) == len(self.segment2_new_coords)):

            self.segment1.mol.vcoords = origin_coords1
            self.segment2.mol.vcoords = origin_coords2

    def stat(self):
        return {
            "amin_sim": round(self.amin_sim, 4),
            "geom_dist": round(self.geom_dist, 4),
        }

class SegmentAlignment(Alignment):
    """
    Detailed icp alignment for 2 segments
    """

    def __init__(self, segment1: "Segment", segment2: "Segment", icp_iterations: int, rotation_list: List[int]):
        """
        :param segment1: first aligned segment
        :param segment2: second aligned segment
        """
        super().__init__(segment1, segment2, icp_iterations, rotation_list)
        self._find_best_alignment()

    def _find_best_alignment(self):
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

        self.icp_alignment(aligned_coords1, aligned_coords2)
        self.compute_amin_sim()

    def draw(self, alpha: float = 0.5):
        Alignment._draw(self, with_whole_surface=False, alpha=alpha)


class MoleculeAlignment(Alignment):
    def __init__(
            self,
            segment1: "Segment",
            segment2: "Segment",
            icp_iterations: int,
            rotations_list: List[int],
            closeness_threshold=CLOSENESS_THRESHOLD
    ):

        super().__init__(segment1, segment2, icp_iterations, rotations_list)
        self.total_amin_sim: Union[float, None] = None
        self.total_geom_dist: Union[float, None] = None
        self.closeness_threshold: float = closeness_threshold
        self._find_best_alignment()
        self.compute_amin_sim()

    @property
    def match_area(self) -> float:

        if not hasattr(self.segment1.mol, 'faces_areas'):
            self.segment1.mol.compute_areas()

        points_dict = {point[0]: point[1] for point in self.correspondence}
        match_area = 0

        for face_idx, points in enumerate(self.segment1.mol.faces):
            close_face = True
            for point in points:
                if (np.linalg.norm(self.segment1.mol.vcoords[point] - self.segment2.mol.vcoords[points_dict[point]])
                        < self.closeness_threshold):
                    close_face = False
            if close_face:
                match_area += self.segment1.mol.faces_areas[face_idx]

        return match_area

    @property
    def match_area_score(self) -> float:
        return self.match_area / (self.segment1.mol.area_of_mesh + self.segment2.mol.area_of_mesh - self.match_area)


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

        self.icp_alignment(aligned_coords1, aligned_coords2)
        self.compute_amin_sim()
        self.total_amin_sim = self.amin_sim * len(self.correspondence)
        self.total_geom_dist = self.geom_dist * len(self.correspondence)

    def draw(self, alpha: float = 0.5):
        Alignment._draw(self, with_whole_surface=True, alpha=alpha)
        
    def stat(self):
        return {
            **super().stat(),
            "total_amin_sim": round(self.total_amin_sim, 4),
            "total_geom_dist": round(self.total_geom_dist, 4),
            "match_area": round(self.match_area, 4),
            "match_area_score": round(self.match_area_score, 4)
        }
