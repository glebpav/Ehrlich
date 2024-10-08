from typing import Iterable, Union, List, Tuple
from pathlib import Path
import pickle
import numpy as np
from naskit.containers.pdb import NucleicAcidChain, ProteinChain, PdbAtom

from .molecule import Molecule
from .mesh import Mesh
from .segment import Segment
from .utils.displacement import Displacement


class MoleculeStructure(Molecule, Mesh):
    """
    Combination of Mesh and Molecule features
    """

    def __init__(
            self,
            anames: Iterable[str],
            acoords: np.ndarray,
            resnum: Iterable[int],
            resnames: Iterable[str],
            molecule: Union[NucleicAcidChain, ProteinChain]
    ):
        Molecule.__init__(self, anames, acoords.astype(np.float32), resnum, resnames)
        Mesh.__init__(self)
        self.molecule: Union[NucleicAcidChain, ProteinChain] = molecule
        self.vamap: List[int] = None  # vertex-atom map: len = number of mesh vertixes. Value - index of closest atom.

    @classmethod
    def from_pdb(cls, pdb: Union[NucleicAcidChain, ProteinChain], center_struct: bool = True) -> "MoleculeStructure":
        """
        Parses chain atoms into arrays and creates MoleculeStructure object
        """

        anames = []
        acoords = []
        resnum = []
        resnames = []

        center_coords = np.array([0., 0., 0.])

        for acid in pdb:
            for atom in acid:
                atom: PdbAtom = atom
                anames.append(atom.name)
                acoords.append(atom.coords)
                resnum.append(atom.moln)
                resnames.append(atom.mol_name)

                if center_struct:
                    center_coords += np.array(atom.coords)

        acoords = np.array(acoords)

        if center_struct:
            center_coords /= len(acoords)
            acoords -= center_coords
            pdb.coords = acoords

        return cls(anames, acoords, resnum, resnames, pdb)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MoleculeStructure":
        """
        Loads pickle file and returns object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: Union[str, Path]):
        """
        Saves self into pickle file
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        return {
            "anames": self.anames,
            "acoords": self.acoords,
            "resnum": self.resnum,
            "resnames": self.resnames,
            "vcoords": self.vcoords,
            "neibs": self.neibs,
            "faces": self.faces,
            "segments": self.segments,
            "vamap": self.vamap,
            "molecule": self.molecule
        }

    def __setstate__(self, d):
        self.anames: Iterable[str] = d.get("anames")
        self.acoords: np.ndarray = d.get("acoords").astype(np.float32)
        self.resnum: Iterable[int] = d.get("resnum")
        self.resnames: Iterable[str] = d.get("resnames")
        self.vcoords: np.ndarray = d.get("vcoords").astype(np.float32)
        self.neibs: List[Tuple[int]] = d.get("neibs")
        self.faces: List[Tuple[int, int, int]] = d.get("faces")
        self.segments: List[Segment] = d.get("segments")
        self.vamap: List[int] = d.get("vamap")
        self.molecule: Union[NucleicAcidChain, ProteinChain] = d.get("molecule")

    def make_mesh(
            self,
            poly_area: float = 25,
            path_to_pdb: str = None,
            path_to_pdb2pqr: str = 'pdb2pqr',
            center_struct: bool = True,
            d: float = 0.6
    ):
        super(MoleculeStructure, self).make_mesh(poly_area, path_to_pdb, path_to_pdb2pqr, center_struct, d=d)
        self.project()

    def project(self):
        """
        Finds closest atom to each mesh vertex and writes to vamap field
        """

        d = np.linalg.norm((self.vcoords[:, np.newaxis, :] - self.acoords), axis=2)
        self.vamap = np.argmin(d, axis=1)

    def get_transform_molecule(self, displacement_list: List[Displacement]) -> Union[NucleicAcidChain, ProteinChain]:
        molecule_copy = self.molecule.copy()
        coords = molecule_copy.coords
        for displacement in displacement_list:
            coords = displacement.displace(coords=coords)
        molecule_copy.coords = coords
        return molecule_copy

