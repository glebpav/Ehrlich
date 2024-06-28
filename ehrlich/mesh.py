import os
import shutil
import sys
from typing import Iterable, Union, List, Tuple
import numpy as np

if sys.version_info >= (3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources

from .segment import Segment



class Mesh:
    
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

    def make_mesh(self, poly_area: float = 25, path_to_pdb: str = None):
        """
        Creates mesh based on molecule atoms and coords. Fills all class fields. 
        Remeshes automaticaly to fit palygon area.
        
        :param poly_area: target area of a triangle polygon.
        """

        d = 0.6
        e = 0.99

        path_to_pqr = path_to_pdb.replace(".pdb", ".pqr")
        for str_item in path_to_pdb.split(".")[0:-1]:
            path_to_pqr += str(str_item)

        # converting pdb to pqr format
        os.system(f"pdb2pqr --ff=AMBER {path_to_pdb} {path_to_pqr}.pqr")

        tms_mesh_pkg = pkg_resources.files("ehrlich")
        tms_mesh_path = tms_mesh_pkg.joinpath("TMSmesh2.1")
        p1_path = tms_mesh_pkg.joinpath("p1.txt")
        p2_path = tms_mesh_pkg.joinpath("p2.txt")
        leg_path = tms_mesh_pkg.joinpath("leg.dat")

        shutil.copyfile(p1_path, "p1.txt")
        shutil.copyfile(p2_path, "p2.txt")
        shutil.copyfile(leg_path, "leg.dat")

        # creating shrunk surface
        os.system(f"{tms_mesh_path} {path_to_pqr}.pqr {d} {e}")

        file_name = f"{path_to_pqr}.pqr-{d}_{e}.off_modify.off"
        with open(file_name) as f:
            lines = f.readlines()

        self.vcoords = np.array([list(map(float, lines[i].split())) for i in range(2, int(lines[1].split()[0]) + 2)])
        print("from here")
        print([list(map(int, lines[i].split()[1:])) for i in range(int(lines[1].split()[0]) + 2, len(lines))])
        # self.neibs = np.array(
        #     [list(map(int, lines[i].split()[1:])) for i in range(int(lines[1].split()[0]) + 2, len(lines))])

        os.remove('p1.txt')
        os.remove('p2.txt')
        os.remove('leg.dat')

        os.remove(file_name)
        os.remove(f"{path_to_pqr}.pqr")
        """
        # converting connections from triangles to adj list
        adj = [set() for _ in range(len(coords))]
        for line in connections:
            idx1, idx2, idx3 = line
            adj[idx1].update([idx2, idx3])
            adj[idx2].update([idx1, idx3])
            adj[idx3].update([idx1, idx2])

        faces = np.array(connections)

        points = []
        for point_idx in range(len(coords)):
            points.append(Point(
                origin_coords=[0, 0, 0],
                shrunk_coords=coords[point_idx],
                neighbors_points_idx=adj[point_idx],
            ))

        # creating object of MoleculeSurface
        molecule_surface = MoleculeSurface(points, [tuple(s) for s in adj])
        molecule_surface.molecule = molecule
        molecule_surface.faces = faces

        molecule_surface = molecule_surface.get_fixed_version()

        return molecule_surface"""
        
        
    def make_segments(area: float = 225):
        """
        Samples vertixes using 'sample' method, creates Segments, 
        calls 'expand' on segments until target area is reached.
        Assigns list of segments to object field.
        """
        ...
        
        
    def sample(self, n: Union[float, int]) -> List[int]:
        """
        Evenly samples vertixes of mesh.
        
        :param n: if float - portion of vertixes, if int - exact number of vertixes
        """
        ...