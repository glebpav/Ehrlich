from typing import Iterable, Union
from pathlib import Path
import pickle
import numpy as np
from naskit.containers.pdb import NucleicAcidChain, ProteinChain, PdbAtom

from .molecule import Molecule
from .mesh import Mesh



class MoleculeStructure(Molecule, Mesh):
    
    def __init__(
                self,
                anames: Iterable[str],
                acoords: np.ndarray, 
                resnum: Iterable[int],
                resnames: Iterable[str]
                ):
        
        Molecule.__init__(self, anames, acoords, resnum, resnames)
        vamap: List[int] = None # vertex-atom map: len = number of mesh vertixes. Value - index of closest atom.
        
        
    @classmethod
    def from_pdb(cls, pdb: Union[NucleicAcidChain, ProteinChain]) -> "MoleculeStructure":
        """
        Parses chain atoms into arrays and creates MoleculeStructure object
        """

        anames = []
        acoords = []
        resnum = []
        resnames = []

        for acid in pdb:
            for atom in acid:
                atom: PdbAtom = atom
                anames.append(atom.name)
                acoords.append(atom.coords)
                resnum.append(atom.moln)
                resnames.append(atom.mol_name)

        acoords = np.array(acoords)
        
        return cls(anames, acoords, resnum, resnames)
        
        
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
            "anames":anames, "acoords":acoords, "resnum":resnum, "resnames":resnames,
            "vcoords":vcoords, "neibs":neibs, "faces":faces, "segments":segments,
            "vamap":vamap
        }
    
    
    def __setstate__(self, d):
        self.anames = d.get("anames")
        self.acoords = d.get("acoords")
        self.resnum = d.get("resnum")
        self.resnames = d.get("resnames")
        self.vcoords = d.get("vcoords")
        self.neibs = d.get("neibs")
        self.faces = d.get("faces")
        self.segments = d.get("segments")
        self.vamap = d.get("vamap")
        
        
    def project(self):
        """
        Finds closest atom to each mesh vertex and writes to vamap field
        """
        ...
    