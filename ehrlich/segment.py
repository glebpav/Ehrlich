import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

from functools import cached_property

from ehrlich import MoleculeSurface, get_rotated_vector, find_inside_cone_points, Point, area_of_triangle, color_list


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
            new_edges, new_points = get_neighbour_data(self.envs_points[-1], self.edge_list, self.surface.faces,
                                                       self.used_points, self.used_edges)
            self.area += self.area_of_faces(new_edges)

            if len(new_edges) * len(new_points) == 0:
                return

            self.used_points += new_points
            self.used_edges += new_edges
            self.envs_surfaces.append(new_edges)
            self.envs_points.append(new_points)

            if area is not None:
                if area < self.area:
                    return

            if max_env is not None:
                if max_env >= len(self.envs_points):
                    return

    @cached_property
    def amines(self):
        used_idxs = {}
        segment_counter = {}
        for env in self.envs_points:
            for point in env:
                acid = self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue
                if acid not in segment_counter:
                    segment_counter[acid] = 1
                    used_idxs[acid] = [self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue_num]
                elif self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue_num not in used_idxs[acid]:
                    segment_counter[acid] += 1
                    used_idxs[acid].append(self.surface.molecule.atoms[self.surface.points[point].atom_idx].residue_num)
        return segment_counter

    def amin_similarity(self, counter2):
        counter_result = {}
        score = 0

        for acid, count in self.amines.items():
            if acid in list(counter2.keys()):
                counter_result[acid] = min(count, counter2[acid])
                score += counter_result[acid]
        return score, counter_result

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