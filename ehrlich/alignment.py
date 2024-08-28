import math
import sys
from abc import ABC
from typing import Union, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from naskit.containers.pdb import NucleicAcidChain, ProteinChain

from ehrlich.utils.amin_similarity import get_amin_idx, amin_similarity_matrix
from ehrlich.utils.displacement import Displacement, Translation, Rotation, Transpose
from ehrlich.utils.icp_helper import icp_step

CLOSENESS_THRESHOLD = 1  # in Angstrom


class Alignment(ABC):
    def __init__(
            self,
            segment1: "Segment",
            segment2: "Segment",
            icp_iterations: int,
            rotations_list: List[int]
    ):

        from .segment import Segment

        self.segment1: Segment = segment1
        self.segment2: Segment = segment2
        self.segment1_new_coords: Union[np.ndarray, None] = None
        self.segment2_new_coords: Union[np.ndarray, None] = None
        self.correspondence: Union[
            List[Tuple[int, int]], None] = None  # [List[Tuple[idx_1_segment, idx_2_segment]] - len(segmant1.mol)
        self.correspondence2: Union[
            np.ndarray, None] = None  # [List[Tuple[idx_2_segment, idx_1_segment]] - len(segmant2.mol)
        self.amin_sim: Union[float, None] = None
        self.geom_dist: Union[float, None] = None
        self.icp_iterations = icp_iterations

        if rotations_list is None:
            rotations_list = [0]

        self.rotations_list = rotations_list

    def z_axis_alignment(
            self,
            aligning_points: np.ndarray,
            origin_point_coords: np.ndarray,
            origin_norm: np.ndarray
    ) -> (np.ndarray, List[Displacement]):

        """
        Compute moved and rotated coords to make origin point in (0; 0; 0) and its norm (0; 0; 1)
        :param aligning_points: np.ndarray with shape (N, 3) of points coordinates that should be aligned
        :param origin_point_coords: np.ndarray with shape (1, 3) of origin point coordinates (after this function it will
        have (0, 0, 0) coordinates)
        :param origin_norm: np.ndarray with shape (1, 3) of normal for origin point
        :return: np.ndarray with shape (N, 3) of aligned points coordinates:
        """

        e3 = origin_norm
        g = np.array([10, 0, -(e3[0] / e3[2])])
        e1 = g / np.linalg.norm(g)
        e2 = np.cross(e3, e1)

        t = np.array([e1, e2, e3]).T
        t_inv = np.linalg.inv(t)
        out_coords = np.dot((aligning_points - origin_point_coords), t_inv.T)

        displacement_queue = [Translation(-origin_point_coords), Rotation(t_inv.T)]

        return np.array(out_coords), displacement_queue

    def compute_amin_sim(self):

        """
        computing amin similarity between two aligned parts
        """

        amin_score = 0.
        for idx1, idx2 in self.correspondence:
            acid_idx1 = get_amin_idx(self.segment1.mol.resnames[self.segment1.mol.vamap[idx1]])
            acid_idx2 = get_amin_idx(self.segment2.mol.resnames[self.segment2.mol.vamap[idx2]])
            amin_score += amin_similarity_matrix[acid_idx1][acid_idx2]

        self.amin_sim = amin_score / len(self.correspondence)

    def icp_alignment(
            self,
            coords1: np.ndarray,
            coords2: np.ndarray,
    ) -> List[Displacement]:

        """
        Finds best position for coords1 using ICP algorithm with the best **rotation** and align
        :param coords1: np.ndarray of coordinates of first aligning part
        :param coords2: np.ndarray of coordinates of second aligning part
        """

        min_norm_value = sys.float_info.max
        out_corresp = None
        out_corresp2 = None
        out_coords = None  # coords of segment1 in best icp alignment for `best_rotated_coords`
        total_displacement_queue = None

        for rotation_idx, rotation_angle in enumerate(self.rotations_list):
            print(f"rotation_idx: {rotation_idx}")

            cos = math.cos(math.radians(rotation_angle))
            sin = math.sin(math.radians(rotation_angle))

            rotated_coords = coords1 @ np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

            coords, norm_values, corresp_values, corresp_values2, displacement_list = self.icp_optimization(
                rotated_coords, coords2, self.icp_iterations
            )

            if min_norm_value > norm_values:
                min_norm_value = norm_values
                out_coords = coords
                out_corresp = corresp_values
                out_corresp2 = corresp_values2
                total_displacement_queue = displacement_list

        self.segment2_new_coords = coords2
        self.segment1_new_coords = out_coords
        self.correspondence = out_corresp
        self.correspondence2 = out_corresp2
        self.geom_dist = min_norm_value

        return total_displacement_queue

    def icp_optimization(
            self,
            coords1: np.ndarray,
            coords2: np.ndarray,
            iterations: int
    ) -> (np.ndarray, float, np.ndarray, np.ndarray, List[Displacement]):

        """
        Find best position for coords1 using ICP algorithm with best align
        :param coords1: np.ndarray of coordinates of first aligning part
        :param coords2: np.ndarray of coordinates of second aligning part
        :param iterations: number of ICP iterations (thus real number of computed iterations could be less in case if
        _icp_break_point() is overridden in child class)
        """

        p_coords = coords1.T
        q_coords = coords2.T

        norm_value = 0.
        p_coords_copy = p_coords.copy()
        correspondence, correspondence2 = None, None
        displacement_queue = [Transpose()]

        for i in range(iterations):
            print(f"icp iteration {i}")
            new_p_coords_copy, new_norm_value, new_correspondence, new_correspondence2, displacement = icp_step(
                p_coords_copy, q_coords)

            if self._icp_break_point(new_p_coords_copy.T, coords2, new_correspondence, new_correspondence2):
                break

            p_coords_copy = new_p_coords_copy
            norm_value = new_norm_value
            correspondence = new_correspondence
            correspondence2 = new_correspondence2
            displacement_queue.append(displacement)

        displacement_queue.append(Transpose())
        return p_coords_copy.T, norm_value, correspondence, correspondence2, displacement_queue

    def _icp_break_point(
            self,
            coords1: np.ndarray,
            coords2: np.ndarray,
            correspondence: np.ndarray,
            correspondence2: np.ndarray
    ) -> bool:
        """
        Function that stops icp algorithm. In base case there is no need to stop by any metric
        only count of iterations important
        :param coords1: np.ndarray of coordinates of first aligning part
        :param coords2: np.ndarray of coordinates of second aligning part
        :param correspondence: np.ndarray of correspondence for each point from first part
        :param correspondence2: np.ndarray of correspondence for each point from second part
        :return: True if icp algorithm should be stopped, False otherwise (in base realization always returns False)
        """
        return False

    def _draw(
            self,
            with_whole_surface: bool,
            alpha: float,
            colored_faces_segment1: List[int] = None,
            colored_faces_segment2: List[int] = None
    ):
        """
        Draw aligned segments using matplotlib
        :param with_whole_surface: bool
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
        self.segment1.draw(
            ax=ax,
            with_whole_surface=with_whole_surface,
            segment_alpha=alpha,
            color="red",
            colored_faces=colored_faces_segment1
        )
        self.segment2.draw(
            ax=ax,
            with_whole_surface=with_whole_surface,
            segment_alpha=alpha,
            color="blue",
            colored_faces=colored_faces_segment2
        )
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

        return ax

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

        aligned_coords1, _ = self.z_axis_alignment(
            np.array([self.segment1.mol.vcoords[point_idx] for point_idx in self.segment1.used_points]),
            self.segment1.mol.vcoords[self.segment1.origin_idx],
            self.segment1.mol.compute_norm(self.segment1.origin_idx)
        )
        aligned_coords2, _ = self.z_axis_alignment(
            np.array([self.segment2.mol.vcoords[point_idx] for point_idx in self.segment2.used_points]),
            self.segment2.mol.vcoords[self.segment2.origin_idx],
            self.segment2.mol.compute_norm(self.segment2.origin_idx)
        )

        _ = self.icp_alignment(aligned_coords1, aligned_coords2)
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
            closeness_threshold: float = CLOSENESS_THRESHOLD
    ):

        super().__init__(segment1, segment2, icp_iterations, rotations_list)
        self.total_amin_sim: Union[float, None] = None
        self.total_geom_dist: Union[float, None] = None
        self.matching_segments1: Union[List[int], None] = None
        self.matching_segments2: Union[List[int], None] = None
        self.closeness_threshold: float = closeness_threshold
        self.displacement_queue1: List[Displacement] = []
        self.displacement_queue2: List[Displacement] = []

        self._find_best_alignment()
        self.compute_amin_sim()

    @property
    def match_area(self) -> float:

        if not hasattr(self.segment1.mol, 'faces_areas'):
            self.segment1.mol.compute_areas()

        if not hasattr(self.segment2.mol, 'faces_areas'):
            self.segment2.mol.compute_areas()

        match_area_segment1 = np.sum(self.segment1.mol.faces_areas[self.matching_segments1])
        match_area_segment2 = np.sum(self.segment2.mol.faces_areas[self.matching_segments2])

        print(f"{match_area_segment1=}")
        print(f"{match_area_segment2=}")

        return (match_area_segment1 + match_area_segment2) / 2

    @property
    def match_area_score(self) -> float:
        return self.match_area / (self.segment1.mol.area_of_mesh + self.segment2.mol.area_of_mesh - self.match_area)

    def _find_best_alignment(self):
        (aligned_coords1, displacement_queue1) = self.z_axis_alignment(
            self.segment1.mol.vcoords,
            self.segment1.mol.vcoords[self.segment1.origin_idx],
            self.segment1.mol.compute_norm(self.segment1.origin_idx)
        )
        aligned_coords2, displacement_queue2 = self.z_axis_alignment(
            self.segment2.mol.vcoords,
            self.segment2.mol.vcoords[self.segment2.origin_idx],
            self.segment2.mol.compute_norm(self.segment2.origin_idx)
        )

        icp_displacement = self.icp_alignment(aligned_coords1, aligned_coords2)

        displacement_queue1 += icp_displacement

        self.compute_amin_sim()
        self.total_amin_sim = self.amin_sim * len(self.correspondence)
        self.total_geom_dist = self.geom_dist * len(self.correspondence)

        self._find_matching_faces()

        self.displacement_queue1 = displacement_queue1
        self.displacement_queue2 = displacement_queue2

    def _find_matching_faces(self):

        self.matching_segments1, self.matching_segments2 = [], []

        points_dict1 = {point[0]: pair_idx for pair_idx, point in enumerate(self.correspondence)}
        points_dict2 = {point[0]: pair_idx for pair_idx, point in enumerate(self.correspondence2)}

        is_pair_close1 = np.linalg.norm(
            self.segment1_new_coords[self.correspondence[:, 0]] - self.segment2_new_coords[self.correspondence[:, 1]],
            axis=1
        ) < self.closeness_threshold

        is_pair_close2 = np.linalg.norm(
            self.segment1_new_coords[self.correspondence2[:, 1]] - self.segment2_new_coords[self.correspondence2[:, 0]],
            axis=1
        ) < self.closeness_threshold

        faces_array = np.array(self.segment1.mol.faces, dtype=object)
        mapped_indices = np.vectorize(lambda x: points_dict1.get(x, -1))(faces_array)
        valid_points_mask = mapped_indices != None
        mapped_indices = np.where(valid_points_mask, mapped_indices, -1).astype(int)

        is_face_close_mask = np.all(valid_points_mask, axis=1) & np.all(
            np.take(is_pair_close1, mapped_indices, axis=0), axis=1
        )

        self.matching_segments1.extend(np.where(is_face_close_mask)[0].tolist())

        faces_array = np.array(self.segment2.mol.faces)

        mapped_indices = np.vectorize(lambda x: points_dict2.get(x, -1))(faces_array)
        valid_points_mask = mapped_indices != None
        mapped_indices = np.where(valid_points_mask, mapped_indices, -1).astype(int)

        is_face_close_mask = np.all(valid_points_mask, axis=1) & np.all(
            np.take(is_pair_close2, mapped_indices, axis=0), axis=1
        )

        self.matching_segments2.extend(np.where(is_face_close_mask)[0].tolist())

    def _icp_break_point(
            self,
            coords1: np.ndarray,
            coords2: np.ndarray,
            correspondence: np.ndarray,
            correspondence2: np.ndarray
    ) -> bool:

        previous_segment1_new_coords = self.segment1_new_coords
        previous_segment2_new_coords = self.segment2_new_coords
        previous_correspondence = self.correspondence
        previous_correspondence2 = self.correspondence2

        self.segment1_new_coords = coords1
        self.segment2_new_coords = coords2
        self.correspondence = correspondence
        self.correspondence2 = correspondence2

        self._find_matching_faces()
        this_icp_score = self.match_area

        if not hasattr(self, "previous_icp_score"):
            self.previous_icp_score = this_icp_score
            return False

        is_smaller = self.previous_icp_score > this_icp_score
        self.previous_icp_score = this_icp_score

        self.segment1_new_coords = previous_segment1_new_coords
        self.segment2_new_coords = previous_segment2_new_coords
        self.correspondence = previous_correspondence
        self.correspondence2 = previous_correspondence2

        return is_smaller

    def get_transformed_molecules(self) -> (
            Union[NucleicAcidChain, ProteinChain],
            Union[NucleicAcidChain, ProteinChain]
    ):

        """
        Returns the transformed molecules.
        First returning molecule is molecule that was gotten from `MoleculeStructure`
        object witch segment had called `mal_align` method, second molecule is that
        you've passed as an argument for `mol_align` method.
        """

        molecule_copy1 = self.segment1.mol.molecule.copy()
        molecule_copy2 = self.segment2.mol.molecule.copy()
        coords1 = molecule_copy1.coords
        coords2 = molecule_copy2.coords
        for displacement in self.displacement_queue1:
            coords1 = displacement.displace(coords=coords1)
        for displacement in self.displacement_queue2:
            coords2 = displacement.displace(coords=coords2)
        molecule_copy1.coords = coords1
        molecule_copy2.coords = coords2
        return molecule_copy1, molecule_copy2

    def draw(self, alpha: float = 0.5):
        ax = Alignment._draw(
            self,
            with_whole_surface=True,
            alpha=alpha,
            colored_faces_segment1=self.matching_segments1,
            colored_faces_segment2=self.matching_segments2
        )
        return ax

    def stat(self):
        return {
            **super().stat(),
            "total_amin_sim": round(self.total_amin_sim, 4),
            "total_geom_dist": round(self.total_geom_dist, 4),
            "match_area": round(self.match_area, 4),
            "match_area_score": round(self.match_area_score, 4)
        }
