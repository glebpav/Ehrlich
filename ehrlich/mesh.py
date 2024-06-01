from typing import Iterable, Union, List, Tuple
import numpy as np

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
        
        
    def make_mesh(poly_area: float = 25):
        """
        Creates mesh based on molecule atoms and coords. Fills all class fields. 
        Remeshes automaticaly to fit palygon area.
        
        :param poly_area: target area of a triangle polygon.
        """
        ...
        
        
    def make_segments(area: float = 225):
        """
        
        """
        ...