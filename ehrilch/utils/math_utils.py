import decimal
import math

import numpy as np


# dist between two points
def get_dist(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2 + (point_a[2] - point_b[2]) ** 2) ** 0.5


# normalizing vector
def get_norm(vect):
    return np.array(vect) / ((vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2) ** 0.5)


# calculating cos between two vectors
def get_cos(vect1, vect2):
    a = vect1[0] * vect2[0] + vect1[1] * vect2[1] + vect1[2] * vect2[2]
    b = np.linalg.norm(vect1) * np.linalg.norm(vect2)
    if b == 0:
        return 0
    return a / b


# calculating geometric center of sphere
def get_center_coords(points):
    center_coords = [0, 0, 0]
    for point in points:
        center_coords += point.origin_coords
    center_coords /= len(points)
    return center_coords


def double_range(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


# calculating rotation matrix by angle and axis
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
def get_rotated_vector(vector, angle_in_degree, axis):
    return np.dot(vector, get_rotation_matrix(angle_in_degree, axis))


# central_angle in degrees
def find_inside_cone_points(points, center_vector, central_angle):
    center_coords = np.array([0., 0., 0.])
    # center_vector = points[central_point_idx].origin_coords
    min_cos = math.cos(math.radians(central_angle))

    list_inside_points_idxs = []
    for point_idx, point in enumerate(points):
        new_point_vector = point.origin_coords - center_coords
        cos = get_cos(center_vector, new_point_vector)

        if cos >= min_cos:
            list_inside_points_idxs.append(point_idx)

    return list_inside_points_idxs
