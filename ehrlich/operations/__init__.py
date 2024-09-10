import ehrlich.operations._find_closest
import numpy as np
import time
def find_closest_points(coords1: np.array, coords2: np.array) -> np.array:
    p = coords1.copy().astype(np.float32)
    q = coords2.copy().astype(np.float32)
    I, D = _find_closest.find(p, q)
    return I, D

