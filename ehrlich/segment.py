from __future__ import annotations

import random
import sys
from functools import cached_property
from typing import Iterable, Union, List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d

# from ehrlich.molecule_structure import MoleculeStructure

from ehrlich.utils.amin_similarity import amino_acid_list, get_amin_idx, amin_similarity_matrix
from ehrlich.utils.icp_helper import icp_optimization
from ehrlich.utils.math_utils import get_rotated_vector, area_of_triangle
from ehrlich.utils.visualize_utils import color_list


class Segment:
    
    def __init__(self, mol: "MoleculeStructure", origin_idx: int):
        """
        :param mol: MoleculeStructure object this segment was built on
        :param origin_idx: index of origin vertex
        envs: list of list of env indices
        area: area of segment
        amins_count: vector of int32 with 20 counters for each aminoacid. Counts all unique amins in segment
        """
        
        self.mol = mol
        self.origin_idx = origin_idx
        self.envs: List[List[int]] = []
        self.area: float = 0.
        self.amins_count: Union[np.ndarray | None] = None
        self.envs_surfaces: Union[np.ndarray | None] = None

        self.d = None

        self.used_points = []
        self.used_faces = []

        self.face_list = [set() for _ in range(len(self.mol.vcoords))]
        for face_idx, face in enumerate(self.mol.faces):
            for point_idx in face:
                self.face_list[point_idx].add(face_idx)
        
    def add_env(self):
        """
        Adds one more env to segment.
        """

        new_faces, new_points = _get_neighbour_data(self.envs[-1], self.face_list, self.mol.faces,
                                                    self.used_points, self.used_faces)
        self.area += self._area_of_faces(new_faces)

        if len(new_faces) * len(new_points) == 0:
            return

        self.used_points += new_points
        self.used_faces += new_faces
        self.face_list.append(new_faces)
        self.envs_surfaces.append(new_faces)
        self.envs.append(list(new_points))

    def expand(self, area: float, max_envs: int = None):
        """
        Adds envs until target area or maximum number of envs is reached.
        
        :param area: target area of segment
        :param max_envs: maximum allowed number of envs in segment
        """

        has_next_env = True
        self.area = 0.
        self.envs = [[self.origin_idx]]
        self.envs_surfaces = []
        self.used_points = [self.origin_idx]
        self.used_faces = []

        while has_next_env:

            self.add_env()

            if area is not None:
                if area < self.area:
                    break

            if max_envs is not None:
                if max_envs >= len(self.envs):
                    break

        self.amins_count = self._compute_amines()

    def amin_similarity(self, segment2: "Segment") -> float:
        score: float = 0.

        counter1 = self.amins_count.copy()
        counter2 = segment2.amins_count.copy()

        for acid_idx in range(len(counter1)):
            addition_score = min(counter1[acid_idx], counter2[acid_idx])
            score += addition_score
            counter1[acid_idx] -= addition_score
            counter2[acid_idx] -= addition_score

        counter_not_empty = True
        while counter_not_empty:
            acid_idx = np.argmax(counter1)
            best_acid = np.argmax(np.where(counter2 > 0, amin_similarity_matrix[acid_idx], 0))
            acid_count = min(counter1[best_acid], counter2[best_acid])
            counter1[best_acid] -= acid_count
            counter2[best_acid] -= acid_count
            score += amin_similarity_matrix[acid_idx][best_acid] * acid_count
            if np.sum(counter1[best_acid]) == 0 or np.sum(counter2[best_acid]) == 0:
                counter_not_empty = False

        score /= np.sum(self.amins_count) + np.sum(segment2.amins_count)
        return score

    def align_compare(self, other_segment) -> SegmentAlignment:

        segment_alignment = SegmentAlignment(self, other_segment)

        aligned_coords1 = self._get_aligned_coords()
        aligned_coords2 = other_segment._get_aligned_coords()

        min_norm_value = sys.float_info.max
        out_corresp = None
        out_coords = None

        for rotation_idx, rotation_angle in enumerate([0, 60, 120, 180, 270]):
            print(f"rotation_idx: {rotation_idx}")
            rotated_coords = np.array([get_rotated_vector(vector, rotation_angle, "z") for vector in aligned_coords1])
            coords, norm_values, corresp_values = icp_optimization(aligned_coords2, rotated_coords)
            if min_norm_value > norm_values:
                min_norm_value = norm_values
                out_coords = coords
                out_corresp = corresp_values

        amin_score = 0.
        for idx1, idx2 in out_corresp:
            acid_idx1 = get_amin_idx(self.mol.resnames[self.mol.vamap[idx1]])
            acid_idx2 = get_amin_idx(self.mol.resnames[self.mol.vamap[idx2]])
            amin_score += amin_similarity_matrix[acid_idx1][acid_idx2]

        segment_alignment.segment2_new_coords = out_coords
        segment_alignment.amin_score = amin_score / len(out_corresp)
        segment_alignment.correspondence = out_coords
        segment_alignment.norm_dist = min_norm_value

        return segment_alignment

    def show(self, with_whole_surface: bool = False, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        ax_max = [0, 0, 0]
        ax_min = [0, 0, 0]

        colors = []
        if with_whole_surface:
            colors = ['grey'] * len(self.mol.faces)
        else:
            colors = ['gray'] * len(self.used_faces)
        vert = []
        for face_idx, face in enumerate(self.mol.faces):
            if not with_whole_surface and face_idx not in self.used_faces:
                continue

            triple = []
            for point in face:
                for i in range(3):
                    ax_min[i] = min(ax_min[i], self.mol.vcoords[point][i])
                    ax_max[i] = max(ax_max[i], self.mol.vcoords[point][i])
                triple.append(self.mol.vcoords[point])
            vert.append(triple)

        ax.set_xlim(ax_min[0], ax_max[0])
        ax.set_ylim(ax_min[1], ax_max[1])
        ax.set_zlim(ax_min[2], ax_max[2])

        max_color_idx = 10
        selected_color = color_list[random.randint(0, max_color_idx)]

        if with_whole_surface:
            for face_idx, face in enumerate(self.mol.faces):
                for level_idx, env_level in enumerate(self.envs_surfaces):
                    if face_idx in env_level:
                        colors[face_idx] = selected_color
        else:
            for idx in range(len(colors)):
                colors[idx] = selected_color

        pc = art3d.Poly3DCollection(vert, facecolors=colors, edgecolor="black")
        ax.add_collection(pc)
        # plt.show()
        return ax

    @cached_property
    def concavity(self) -> float:
        # todo: full refactor
        norm_vect = self.mol.compute_norm(self.origin_idx)
        m = []
        for level in self.envs[1:]:
            for point_idx in level:
                vect = (self.mol.vcoords[point_idx] - self.mol.vcoords[self.origin_idx])
                vect /= np.linalg.norm(vect)
                m.append(vect)

        m = np.array(m)
        self.d = norm_vect @ m.T

        return np.mean(self.d)

    @cached_property
    def curvature(self) -> float:
        # todo: full refactor
        if self.d is None: self.concavity
        return self.d @ self.d

    def _compute_amines(self) -> np.ndarray:
        used_idxs = [[] for _ in range(len(amino_acid_list))]
        segment_counter = np.zeros(len(amino_acid_list), dtype=int)
        for env_level in self.envs:
            for point in env_level:
                # acid = self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue[0]
                acid = self.mol.resnames[self.mol.vamap[point]]
                # residue_num = self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue_num
                residue_num = self.mol.resnum[self.mol.vamap[point]]
                if residue_num not in used_idxs[get_amin_idx(acid)]:
                    segment_counter[get_amin_idx(acid)] += 1.
                    used_idxs[get_amin_idx(acid)].append(residue_num)
        return segment_counter

    def _area_of_faces(self, used_faces):
        out_area = 0.
        for face in used_faces:
            out_area += area_of_triangle(self.mol.vcoords[self.mol.faces[face][0]],
                                         self.mol.vcoords[self.mol.faces[face][1]],
                                         self.mol.vcoords[self.mol.faces[face][2]])
        return out_area

    def _get_aligned_coords(self) -> np.ndarray:
        center_point_coords = self.mol.vcoords[self.origin_idx]

        e3 = self.mol.compute_norm(self.origin_idx)
        g = np.array([10, 0, -(e3[0] / e3[2])])
        e1 = g / np.linalg.norm(g)
        e2 = np.cross(e3, e1)

        t = np.array([e1, e2, e3]).T
        t_inv = np.linalg.inv(t)

        out_coords = []
        for idx, point_idx in enumerate(self.used_points):
            out_coords.append(np.matmul(t_inv, self.mol.vcoords[point_idx] - center_point_coords))

        return np.array(out_coords)


class SegmentAlignment:
    def __init__(self, segment1: Segment, segment2: Segment):
        self.segment1 = segment1
        self.segment2 = segment2
        self.segment2_new_coords: Union[np.ndarray | None] = None
        self.correspondence: Union[List[Tuple[int, int]] | None] = None
        self.amin_sim: Union[float | None] = None
        self.norm_dist: Union[float | None] = None

    def show(self):
        origin_coords1 = self.segment1.mol.vcoords
        origin_coords2 = self.segment2.mol.vcoords
        segment1_aligned_coords = self.segment1._get_aligned_coords()

        for idx, point_idx in enumerate(self.segment2.used_points):
            self.segment2.mol.vcoords[point_idx] = self.segment2_new_coords[idx]
        for idx, point_idx in enumerate(self.segment1.used_points):
            self.segment1.mol.vcoords[point_idx] = segment1_aligned_coords[idx]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        self.segment1.show(ax=ax, with_whole_surface=False)
        self.segment2.show(ax=ax, with_whole_surface=False)
        plt.show()

        for idx, point_idx in enumerate(self.segment2.used_points):
            self.segment2.mol.vcoords[point_idx] = origin_coords2[point_idx]
        for idx, point_idx in enumerate(self.segment1.used_points):
            self.segment1.mol.vcoords[point_idx] = origin_coords1[point_idx]


def _get_neighbour_data(old_points_idxs, faces_list, points_list, used_points, used_faces):
    neig_points, neig_faces = set(), set()
    for point_idx in old_points_idxs:
        filtered_faces = list(filter(lambda item: item not in used_faces, faces_list[point_idx]))
        neig_faces.update(filtered_faces)
        for face_idx in filtered_faces:
            face = points_list[face_idx]
            face = list(filter(lambda item: item not in used_points, face))
            neig_points.update(face)
    return neig_faces, neig_points

