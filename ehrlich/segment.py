import math
import sys
from typing import Dict, Any, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

from functools import cached_property

from numpy import ndarray

from ehrlich import MoleculeSurface, get_rotated_vector, find_inside_cone_points, Point, area_of_triangle, color_list
from ehrlich.amin_similarity import amino_acid_list, get_amin_idx, amin_similarity_matrix
from ehrlich.utils.icp_helper import icp_optimization


class Segment:
    def __init__(self, center_point_idx, surface: MoleculeSurface):
        self.used_edges = []
        self.area = 0.
        self.used_points = None
        self.center_point_idx = center_point_idx
        self.surface = surface
        self.envs_points = []
        self.envs_surfaces = []

        # preparing data
        self.edge_list = [set() for _ in range(len(self.surface.points))]
        for edge_idx, edge in enumerate(self.surface.faces):
            for point_idx in edge:
                self.edge_list[point_idx].add(edge_idx)

    def area_of_faces(self, used_faces):
        out_area = 0.
        for face in used_faces:
            out_area += area_of_triangle(self.surface.points[self.surface.faces[face][0]].shrunk_coords,
                                         self.surface.points[self.surface.faces[face][1]].shrunk_coords,
                                         self.surface.points[self.surface.faces[face][2]].shrunk_coords)
        return out_area

    def add_env(self, n=1):
        for env_idx in range(n):
            new_edges, new_points = get_neighbour_data(self.envs_points[-1], self.edge_list, self.surface.faces,
                                                       self.used_points, self.used_edges)
            self.area += self.area_of_faces(new_edges)

            if len(new_edges) * len(new_points) == 0:
                return

            self.used_points += new_points
            self.used_edges += new_edges
            self.envs_surfaces.append(new_edges)
            self.envs_points.append(new_points)

    def expand(self, area=None, max_env=None):

        # processing
        has_next_env = True
        self.area = 0.
        self.envs_points = [[self.center_point_idx]]
        self.used_points = [self.center_point_idx]
        self.used_edges = []

        while has_next_env:

            self.add_env()

            if area is not None:
                if area < self.area:
                    return

            if max_env is not None:
                if max_env >= len(self.envs_points):
                    return

    def amin_similarity(self, counter2_origin: ndarray) -> float:
        score: float = 0.

        counter1 = self.amines.copy()
        counter2 = counter2_origin.copy()

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

        print(f"{score=}")
        print(f"{counter1=}")
        print(f"{counter2=}")
        print(f"{self.amines=}")
        print(f"{counter2_origin=}")
        print(f"{np.sum(counter2_origin)=}")
        print(f"{np.sum(self.amines)=}")

        score /= np.sum(self.amines) + np.sum(counter2_origin)
        return score

    def show(self, surface):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax_max = [0, 0, 0]
        ax_min = [0, 0, 0]

        colors = ['grey'] * len(surface.faces)
        vert = []
        for face in surface.faces:
            triple = []
            for point in face:
                for i in range(3):
                    if surface.points[point].shrunk_coords[i] > ax_max[i]:
                        ax_max[i] = surface.points[point].shrunk_coords[i]
                    if surface.points[point].shrunk_coords[i] < ax_min[i]:
                        ax_min[i] = surface.points[point].shrunk_coords[i]
                triple.append(surface.points[point].shrunk_coords)
            vert.append(triple)

        ax.set_xlim(ax_min[0], ax_max[0])
        ax.set_ylim(ax_min[1], ax_max[1])
        ax.set_zlim(ax_min[2], ax_max[2])

        for face_idx, face in enumerate(surface.faces):
            for level_idx, env_level in enumerate(self.envs_surfaces):
                if face_idx in env_level:
                    colors[face_idx] = color_list[level_idx]

        pc = art3d.Poly3DCollection(vert, facecolors=colors, edgecolor="black")
        ax.add_collection(pc)

    # Compute coords that:
    # center point moved to (0; 0; 0) and
    # rotated to make norm of segment (0; 0; 1)
    def get_aligned_coords(self) -> np.ndarray:
        center_point = self.surface.points[self.center_point_idx]
        center_point_coords = center_point.shrunk_coords

        e3 = center_point.compute_norm(self.surface)
        g = np.array([10, 0, -(e3[0] / e3[2])])
        e1 = g / np.linalg.norm(g)
        e2 = np.cross(e3, e1)

        t = np.array([e1, e2, e3]).T
        t_inv = np.linalg.inv(t)

        out_coords = []
        for idx, point_idx in enumerate(self.used_points):
            out_coords.append(np.matmul(t_inv, self.surface.points[point_idx].shrunk_coords - center_point_coords))

        return np.array(out_coords)

    def compare(self, other_segment) -> Tuple[float, float]:

        aligned_coords1 = self.get_aligned_coords()
        aligned_coords2 = other_segment.get_aligned_coords()

        min_norm_value = sys.float_info.max
        out_corresp = None
        out_coords = None

        for rotation_idx, rotation_angle in enumerate([0, 60, 120, 180, 270]):
            print(f'ration {rotation_idx + 1} out of {6}')
            rotated_coords = np.array([get_rotated_vector(vector, rotation_angle, "z") for vector in aligned_coords1])
            coords, norm_values, corresp_values = icp_optimization(rotated_coords, aligned_coords2)
            if min_norm_value > norm_values:
                min_norm_value = norm_values
                out_coords = coords
                out_corresp = corresp_values

        amin_score = 0.
        for idx1, idx2 in out_corresp:

            atom_idx1 = self.surface.points[idx1].atom_idx
            atom_idx2 = self.surface.points[idx2].atom_idx

            acid1 = self.surface.molecule.atoms[atom_idx1].residue[0]
            acid2 = self.surface.molecule.atoms[atom_idx2].residue[0]

            acid_idx1 = get_amin_idx(acid1)
            acid_idx2 = get_amin_idx(acid2)

            amin_score += amin_similarity_matrix[acid_idx1][acid_idx2]

        return min_norm_value, amin_score / len(out_corresp)


    @cached_property
    def amines(self) -> ndarray:
        used_idxs = [[] for _ in range(len(amino_acid_list))]
        segment_counter = np.zeros(len(amino_acid_list), dtype=int)
        for env_level in self.envs_points:
            for point in env_level:
                acid = self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue[0]
                residue_num = self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue_num
                print(f"{acid=} {residue_num=}")
                if residue_num not in used_idxs[get_amin_idx(acid)]:
                    segment_counter[get_amin_idx(acid)] += 1.
                    used_idxs[get_amin_idx(acid)].append(residue_num)
        return segment_counter

    @cached_property
    def concavity(self) -> float:
        norm_vect = self.surface.points[self.center_point_idx].compute_norm(self.surface)
        m = []
        for level in self.envs_points[1:]:
            for point_idx in level:

                vect = (self.surface.points[point_idx].shrunk_coords
                        - self.surface.points[self.center_point_idx].shrunk_coords)
                vect /= np.linalg.norm(vect)

                m.append(vect)

        m = np.array(m)
        self.d = norm_vect@m.T

        return np.mean(self.d)

    @cached_property
    def curvature(self) -> float:
        if self.d is None: self.concavity
        return self.d@self.d


def get_neighbour_data(old_points_idxs, edge_list, points_list, used_points, used_edges):
    neig_points, neig_edges = set(), set()
    for point_idx in old_points_idxs:
        filtered_edges = list(filter(lambda item: item not in used_edges, edge_list[point_idx]))
        neig_edges.update(filtered_edges)
        for edge_idx in filtered_edges:
            edge = points_list[edge_idx]
            edge = list(filter(lambda item: item not in used_points, edge))
            neig_points.update(edge)
    return neig_edges, neig_points