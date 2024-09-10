import math
import os
import random
import shutil
import sys
from functools import cached_property
from typing import Iterable, Union, List, Tuple
import numpy as np

from stl import mesh
import pyvista as pv
import pyacvd

from .utils.math_utils import area_of_triangle

if sys.version_info >= (3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources

from .segment import Segment


class Mesh:
    """
    Surface implementation for molecule structure
    """

    def __init__(self):
        """
        :param vcoords: float Nx3 matrix of vertex coordinates
        :param neibs: list of neighbors indices of each vertex
        :param faces: list of indices forming face
        :param segments: list of Segment objects
        """

        self.vcoords: np.ndarray = None
        self.neibs: List[Tuple[int]] = None
        self.faces: List[Tuple[int, int, int]] = None
        self.segments: List[Segment] = None
        self.faces_areas: np.ndarray = None  # List[int]

    def make_mesh(
            self,
            poly_area: float = 25,
            path_to_pdb: str = None,
            path_to_pdb2pqr: str = 'pdb2pqr',
            center_struct: bool = True,
            d: float = 0.6
    ) -> None:
        """
        Creates mesh based on molecule atoms and coords. Fills all class fields. 
        Remeshes automatically to fit polygon area.
        
        :param poly_area: target area of a triangle polygon.
        :param path_to_pdb: path to pdb file
        :param path_to_pdb2pqr: path to pdb2pqr file
        :param center_struct: if true - geometry center of vertices will be in (0; 0; 0)
        :param d: TMSmesh parameter in [0.1 ... 0.9] of surface quality. The higher the
        value, the greater the number of points forming the surface and it is more detailed
        """

        # d = 0.4
        e = 0.99

        path_to_pqr = path_to_pdb.replace(".pdb", "")

        # converting pdb to pqr format
        os.system(f"{path_to_pdb2pqr} --ff=AMBER {path_to_pdb} {path_to_pqr}.pqr")

        tms_mesh_pkg = pkg_resources.files("ehrlich")
        tms_mesh_path = tms_mesh_pkg.joinpath("tmsmesh/TMSmesh2.1")
        p1_path = tms_mesh_pkg.joinpath("tmsmesh/p1.txt")
        p2_path = tms_mesh_pkg.joinpath("tmsmesh/p2.txt")
        leg_path = tms_mesh_pkg.joinpath("tmsmesh/leg.dat")

        shutil.copyfile(p1_path, "p1.txt")
        shutil.copyfile(p2_path, "p2.txt")
        shutil.copyfile(leg_path, "leg.dat")

        # creating shrunk surface
        os.system(f"{tms_mesh_path} {path_to_pqr}.pqr {d} {e}")

        file_name = f"{path_to_pqr}.pqr-{d}_{e}.off_modify.off"
        with open(file_name) as f:
            lines = f.readlines()

        self.vcoords = np.array([list(map(float, lines[i].split())) for i in range(2, int(lines[1].split()[0]) + 2)]).astype(np.float32)
        self.faces = [tuple(map(int, lines[i].split()[1:])) for i in range(int(lines[1].split()[0]) + 2, len(lines))]
        self.neibs = [[None]] * len(self.vcoords)
        neibs = [[-1]] * len(self.vcoords)

        for face in self.faces:
            for idx in face:
                neibs[idx] = neibs[idx].copy() + [i for i in face if i != idx]

        for idx, i in enumerate(neibs):
            a = list(set(i))
            a.sort()
            self.neibs[idx] = tuple(filter(lambda x: x != -1, a))

        os.remove('p1.txt')
        os.remove('p2.txt')
        os.remove('leg.dat')

        os.remove(file_name)
        os.remove(f"{path_to_pqr}.pqr")

        fixed_mesh = self._get_fixed_version()

        self.vcoords = fixed_mesh.vcoords
        self.neibs = fixed_mesh.neibs
        self.faces = fixed_mesh.faces

        if center_struct:
            center_coords = np.array([0., 0., 0.])
            for coord in self.vcoords:
                center_coords += coord
            center_coords /= len(self.vcoords)
            self.vcoords -= center_coords

        self.compute_areas()

    def make_segments(self, area: float = 225):
        """
        Samples vertixes using 'sample' method, creates Segments, 
        calls 'expand' on segments until target area is reached.
        Assigns list of segments to object field.
        """

        segments_number = round(math.pi * (self.area_of_mesh / area))
        # print(f"{segments_number=}")

        v_idxs = self._sample(segments_number)
        segments = []
        for iteration, idx in enumerate(v_idxs):
            # print(f"{iteration} out of {segments_number}")
            segment = Segment(self, idx, iteration)
            segment.expand(area)
            segments.append(segment)

        self.segments = segments

    def _sample(self, n: Union[float, int]) -> List[int]:
        """
        Evenly samples vertixes of mesh.
        
        :param n: if float - portion of vertixes, if int - exact number of vertixes
        """

        if isinstance(n, int):
            pass
        elif isinstance(n, float) and 0 < n <= 1:
            n = int(n * len(self.vcoords))
        else:
            return list(range(len(self.vcoords)))

        # TODO: remove redundant save
        self.to_stl().save('buffer_surface.stl')
        input_mesh = pv.PolyData('buffer_surface.stl')
        clus = pyacvd.Clustering(input_mesh)

        clus.subdivide(1)
        clus.cluster(n)

        output_mesh = clus.create_mesh()
        print(len(output_mesh.points))

        get_dist = lambda x, y: np.linalg.norm(x - y)

        points_idxs = random.sample(list(range(len(output_mesh.points))), min(len(output_mesh.points), n))
        mapped_points = [0] * n
        for output_idx, point_idx in enumerate(points_idxs):
            min_dist = get_dist(output_mesh.points[point_idx], input_mesh.points[0])

            for input_idx, b_coord in enumerate(input_mesh.points):
                new_dist = get_dist(output_mesh.points[point_idx], b_coord)
                if new_dist < min_dist:
                    min_dist = new_dist
                    mapped_points[output_idx] = input_idx

        return mapped_points

    def to_stl(self):
        """
        Prepare surface to save in stl
        just write o.to_stl().save('name.stl') to save in stl format
        :return: object that can be saved
        """

        mesh_object = mesh.Mesh(np.zeros(len(self.faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(self.faces):
            for j in range(3):
                mesh_object.vectors[i][j] = self.vcoords[f[j]]
        return mesh_object

    def compute_norm(self, point_idx: int) -> Union[np.ndarray, None]:
        """
        Computes norm of point for this surface
        :param point_idx: index of point
        :return: norm of point or None in case no such point
        """

        if point_idx == -1 or point_idx >= len(self.vcoords):
            print("Error no such point")
            return None

        env_by_levels = {level_idx: [] for level_idx in range(3)}
        used_points = [point_idx]
        for level_idx in range(3):
            adj_points_idxs = []
            for selected_point_idx in used_points:
                adj_points_idxs += [adj_point_idx for adj_point_idx in self.neibs[selected_point_idx]
                                    if adj_point_idx not in used_points]

            adj_points_idxs = list(set(adj_points_idxs))

            get_dist = lambda x, y: np.linalg.norm(x - y)

            # sorting by cw or ccw order
            for idx1 in range(1, len(adj_points_idxs)):
                for idx2 in range(idx1 + 1, len(adj_points_idxs)):
                    dist1 = get_dist(self.vcoords[adj_points_idxs[idx1 - 1]], self.vcoords[adj_points_idxs[idx2]])
                    dist2 = get_dist(self.vcoords[adj_points_idxs[idx1 - 1]], self.vcoords[adj_points_idxs[idx1]])

                    if dist1 < dist2:
                        temp = adj_points_idxs[idx1]
                        adj_points_idxs[idx1] = adj_points_idxs[idx2]
                        adj_points_idxs[idx2] = temp

            vect1 = self.vcoords[adj_points_idxs[0]] - self.vcoords[point_idx]
            vect2 = self.vcoords[adj_points_idxs[1]] - self.vcoords[point_idx]
            res_vect = np.cross(vect1, vect2)
            center_vector = self.vcoords[point_idx]

            cos_a = np.dot(res_vect, center_vector) / (np.linalg.norm(res_vect) * np.linalg.norm(center_vector))

            # 1 2 3 4 ...
            if cos_a > 0:
                vectors_sequence = list(range(len(adj_points_idxs)))
            # n n-1 n-2 ...
            else:
                vectors_sequence = list(range(len(adj_points_idxs) - 1, -1, -1))

            for idx, vector_idx in enumerate(vectors_sequence):
                env_by_levels[level_idx].append(adj_points_idxs[vectors_sequence[idx]])
            used_points += adj_points_idxs

        norm_components = []
        for i in range(0, 3):
            for idx in range(1, len(env_by_levels[i])):
                vect1 = self.vcoords[env_by_levels[i][idx - 1]] - self.vcoords[point_idx]
                vect2 = self.vcoords[env_by_levels[i][idx]] - self.vcoords[point_idx]
                res_vect = np.cross(vect1, vect2)
                norm_components.append(res_vect / np.linalg.norm(res_vect))

        avg_adj_vect = np.average(np.array(norm_components), axis=0)
        norm_avg = np.array(avg_adj_vect / np.linalg.norm(avg_adj_vect))
        cos = (np.dot(avg_adj_vect, self.vcoords[point_idx])
               / (np.linalg.norm(avg_adj_vect) * np.linalg.norm(self.vcoords[point_idx])))
        is_norm_inverted = cos > 0

        if not is_norm_inverted:
            norm_avg *= -1

        return norm_avg

    def compute_areas(self):
        self.faces_areas = np.array(
            [area_of_triangle(self.vcoords[face[0]], self.vcoords[face[1]], self.vcoords[face[2]]) for face in
             self.faces])

    def _get_fixed_version(self) -> "Mesh":
        """
        Make existing surface smoother by remeshing it
        :return: Mesh object
        """

        self.to_stl().save('buffer_surface.stl')
        input_mesh = pv.PolyData('buffer_surface.stl')
        clus = pyacvd.Clustering(input_mesh)

        clus.subdivide(3)
        clus.cluster(len(self.vcoords) / 2)

        output_mesh = clus.create_mesh()
        output_faces = [list(output_mesh.faces[1 + i * 4: (i + 1) * 4:]) for i in
                        range(int(len(output_mesh.faces) / 4))]

        adj = [set() for _ in range(len(output_mesh.points))]
        for line in output_faces:
            idx1, idx2, idx3 = line
            adj[idx1].update([int(idx2), int(idx3)])
            adj[idx2].update([int(idx1), int(idx3)])
            adj[idx3].update([int(idx1), int(idx2)])

        faces = np.array(output_faces)

        fixed_mesh = Mesh()
        fixed_mesh.vcoords = np.array(output_mesh.points)
        fixed_mesh.neibs = [tuple(neib) for neib in adj]
        fixed_mesh.faces = list(map(tuple, faces))

        return fixed_mesh

    @cached_property
    def area_of_mesh(self) -> float:
        """
        Whole area of mesh's surface computed by sum of faces' area
        """

        out_area = 0.
        for face in self.faces:
            out_area += area_of_triangle(self.vcoords[face[0]],
                                         self.vcoords[face[1]],
                                         self.vcoords[face[2]])
        return out_area
