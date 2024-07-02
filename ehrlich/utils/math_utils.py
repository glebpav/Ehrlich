import math

import numpy as np


def get_rotation_matrix(angle_in_degree, axis):
    cos = math.cos(math.radians(angle_in_degree))
    sin = math.sin(math.radians(angle_in_degree))
    if axis == 'z':
        return np.array([
            [cos, -sin, 0],
            [sin,  cos, 0],
            [0,    0,   1]
        ])
    elif axis == 'x':
        return np.array([
            [1,  0,    0  ],
            [0,  cos, -sin],
            [0,  sin,  cos]
        ])
    elif axis == 'y':
        return np.array([
            [cos,  0, sin],
            [0,    1,   0],
            [-sin, 0, cos]
        ])


# rotate vector by vector and
def get_rotated_vector(vector, angle_in_degree, axis: str):
    return np.dot(vector, get_rotation_matrix(angle_in_degree, axis))