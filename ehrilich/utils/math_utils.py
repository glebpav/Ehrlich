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
