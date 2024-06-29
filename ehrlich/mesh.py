import os
import shutil
import sys
from typing import Iterable, Union, List, Tuple
import numpy as np

from stl import mesh
import pyvista as pv
import pyacvd


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

        path_to_pqr = path_to_pdb.replace(".pdb", "")

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

    def _get_fixed_version(self):
        self.to_stl().save('buffer_surface.stl')
        input_mesh = pv.PolyData('buffer_surface.stl')
        clus = pyacvd.Clustering(input_mesh)

        clus.subdivide(3)
        clus.cluster(len(self.vcoords))

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