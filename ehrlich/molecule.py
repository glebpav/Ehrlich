from typing import Iterable, Union
import numpy as np



class Molecule:
    
    def __init__(
                self, 
                anames: Iterable[str],
                acoords: np.ndarray, 
                resnum: Iterable[int],
                resnames: Iterable[str]
                ):
        
        self.anames = anames
        self.acoords = acoords
        self.resnum = resnum
        self.resnames = resnames
        
        
    # Optional
    # def remove_inner_atoms(self):
    #     """
    #     Removes inner atoms not involved in mesh creation
    #     """