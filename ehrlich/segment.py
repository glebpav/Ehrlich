
from __future__ import annotations
import random
import sys
from functools import cached_property
from typing import Iterable, Union, List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d

from ehrlich.alignment import SegmentAlignment, MoleculeAlignment

from ehrlich.utils.amin_similarity import amino_acid_list, get_amin_idx, amin_similarity_matrix
from ehrlich.utils.math_utils import area_of_triangle
from ehrlich.utils.visualize_utils import color_list


class Segment:
    """
    Circular region of structure surface witch used to make comparisons
    """
    
    def __init__(self, mol: "MoleculeStructure", origin_idx: int):
        """
        :param mol: MoleculeStructure object this segment was built on
        :param origin_idx: index of origin vertex
        envs: list of list of env indices
        area: area of segment
        amins_count: vector of int32 with 20 counters for each aminoacid. Counts all unique amins in segment
        """

        from molecule_structure import MoleculeStructure

        self.mol: MoleculeStructure = mol
        self.origin_idx = origin_idx
        self.envs: List[List[int]] = []
        self.area: float = 0.
        # self.amins_count: Union[np.ndarray | None] = None
        self.envs_surfaces: Union[np.ndarray | None] = None

        self.used_points = []
        self.used_faces = []
        self.d = None
        self.face_list = [set() for _ in range(len(self.mol.vcoords))]
        for face_idx, face in enumerate(self.mol.faces):
            for point_idx in face:
                self.face_list[point_idx].add(face_idx)

    def add_env(self):
        """
        Adds one more env to segment.
        """
        print(f"Adding env {self.area}")
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

        self.envs_surfaces = np.array(self.envs_surfaces)

    def amin_similarity(self, segment2: "Segment") -> float:
        """
        Computes amino acid similarity between two segments.
        :param segment2: Segment to compute amino acid similarity for
        :return: amino acid similarity
        """

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

    def align(self, other_segment, icp_iterations: int = 1, rotation_list: List[int] = None) -> SegmentAlignment:
        """
        Finds best alignment between two segments.
        :param other_segment: Segment to align to
        :param icp_iterations: number of iterations to perform the ICP algorithm
        :param rotation_list: list of rotation angles, in case `None` no rotation will not be applied
        :return: SegmentAlignment object witch holds all align info
        """

        segment_alignment = SegmentAlignment(self, other_segment, icp_iterations, rotation_list)
        return segment_alignment

    def mol_align(self, other_segment: "Segment", icp_iterations: int = 1, rotation_list: List[int] = None) -> MoleculeAlignment:
        """
        Finds best alignment between two molecules by segments.
        :param other_segment: Segment to align to
        :param icp_iterations: number of iterations to perform the ICP algorithm
        :param rotation_list: list of rotation angles, in case `None` no rotation will not be applied
        :return: SegmentAlignment object witch holds all align info
        """

        molecule_alignment = MoleculeAlignment(self, other_segment, icp_iterations, rotation_list)
        return molecule_alignment

    def draw(
            self,
            with_whole_surface: bool = False,
            ax=None,
            segment_alpha: float = 0.5,
            color: str = None,
            colored_faces: List[int] = None):
        """
        Draw segment using plt
        :param with_whole_surface: if true - print colored segment with left gray structure surface / if false - print only colored segment
        :param ax: matplotlib axes object, could be omitted
        :param segment_alpha: alpha value between 0 and 1
        :param color: color to draw the segment, in case `None` random color will be used
        :param colored_faces: list of indices of colored segments, in case `None` only segment faces will be drawn
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        # todo: refactor
        if colored_faces is None:
            if not isinstance(self.envs_surfaces, np.ndarray):
                self.envs_surfaces = np.array([point for env_level in self.envs_surfaces for point in env_level])
            colored_faces = self.envs_surfaces.flatten()

        max_color_idx = 15
        if color is None:
            color = color_list[random.randint(0, max_color_idx)]

        ax_max = [0, 0, 0]
        ax_min = [0, 0, 0]

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

        ax.set_xlim(min(ax_min), max(ax_max))
        ax.set_ylim(min(ax_min), max(ax_max))
        ax.set_zlim(min(ax_min), max(ax_max))

        if with_whole_surface:
            for face_idx, face in enumerate(self.mol.faces):
                if face_idx in colored_faces:
                    colors[face_idx] = color
        else:
            for idx in range(len(colors)):
                colors[idx] = color

        pc = art3d.Poly3DCollection(vert, facecolors=colors, edgecolor=(0., 0., 0., segment_alpha), alpha=segment_alpha)
        ax.add_collection(pc)
        norm = self.mol.compute_norm(self.origin_idx) * 3
        origin = self.mol.vcoords[self.origin_idx]
        plt.quiver(origin[0], origin[1], origin[2], norm[0], norm[1], norm[2], linewidth=3)

        # plt.show()
        return ax

    @cached_property
    def concavity(self) -> float:
        """
        Metric for measuring segment concavity
        """
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
        """
        Metric for measuring segment curvature
        """
        if self.d is None: self.concavity
        return self.d @ self.d

    @cached_property
    def amins_count(self) -> np.ndarray:
        """
        Compute amino acid counters. Count of each
        :return: np.ndarray of all unique amins in segment
        """

        if not hasattr(self.mol, 'vamap'):
            self.mol.project()

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
        """
        Compute area of whole segment
        :param used_faces: np.ndarray of all used faces in segment
        :return: area of whole segment
        """
        out_area = 0.
        for face in used_faces:
            out_area += area_of_triangle(self.mol.vcoords[self.mol.faces[face][0]],
                                         self.mol.vcoords[self.mol.faces[face][1]],
                                         self.mol.vcoords[self.mol.faces[face][2]])
        return out_area


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

