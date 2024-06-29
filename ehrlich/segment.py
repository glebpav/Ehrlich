from typing import Iterable, Union, List, Tuple
import numpy as np


class Segment:
    
    def __init__(
                self, 
                mol, # Molecular_structure
                origin_idx: int
                ):
        """
        :param mol: MoleculeStructure object this segment was built on
        :param origin_idx: index of origin vertex
        :param envs: list of list of env indices
        :param area: area of segment
        :param amins_count: vector of int32 with 20 counters for each aminoacid. Counts all unique amins in segment 
        """
        
        self.mol = mol
        self.origin_idx = origin_idx
        self.envs: List[List[int]] = []
        self.area: float = 0.
        self.amins_count: np.ndarray = None

        self.used_points = []
        self.used_edges = []

        # preparing data
        self.edge_list = [set() for _ in range(len(self.mol.vcoords))]
        for edge_idx, edge in enumerate(self.mol.faces):
            for point_idx in edge:
                self.edge_list[point_idx].add(edge_idx)
        
    def add_env(self):
        """
        Adds one more env to segment.
        """

        new_edges, new_points = _get_neighbour_data(self.envs[-1], self.edge_list, self.mol.faces,
                                                    self.used_points, self.used_edges)
        self.area += self._area_of_faces(new_edges)

        if len(new_edges) * len(new_points) == 0:
            return

        self.used_points += new_points
        self.used_edges += new_edges
        self.edge_list.append(new_edges)
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
        self.used_points = [self.origin_idx]
        self.used_edges = []

        while has_next_env:

            self.add_env()

            if area is not None:
                if area < self.area:
                    return

            if max_envs is not None:
                if max_envs >= len(self.envs):
                    return

    def _area_of_faces(self, used_faces):
        out_area = 0.
        for face in used_faces:
            out_area += _area_of_triangle(self.mol.vcoords[self.mol.faces[face][0]],
                                          self.mol.vcoords[self.mol.faces[face][1]],
                                          self.mol.vcoords[self.mol.faces[face][2]])
        return out_area


def _area_of_triangle(p1, p2, p3):
    v1 = np.array(p1)
    v2 = np.array(p2)
    v3 = np.array(p3)
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


def _get_neighbour_data(old_points_idxs, edge_list, points_list, used_points, used_edges):
    neig_points, neig_edges = set(), set()
    for point_idx in old_points_idxs:
        filtered_edges = list(filter(lambda item: item not in used_edges, edge_list[point_idx]))
        neig_edges.update(filtered_edges)
        for edge_idx in filtered_edges:
            edge = points_list[edge_idx]
            edge = list(filter(lambda item: item not in used_points, edge))
            neig_points.update(edge)
    return neig_edges, neig_points

