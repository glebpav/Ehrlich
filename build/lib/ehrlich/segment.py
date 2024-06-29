from typing import Iterable, Union, List, Tuple
import numpy as np



class Segment:
    
    def __init__(
                self, 
                mol: "MoleculeStructure", 
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
        self.envs: List[List[int]] = None
        self.area: float = 0.
        self.amins_count: np.ndarray = None
        
        
    def add_env(self):
        """
        Adds one more env to segment.
        """
        ...
        
        
    def expand(self, area: float, max_envs: int = None):
        """
        Adds envs until target area or maximum number of envs is reached.
        
        :param area: target area of segment
        :param max_envs: maximum allowed number of envs in segment
        """
        ...