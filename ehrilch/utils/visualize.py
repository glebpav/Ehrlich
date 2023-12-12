import math

import numpy as np

from ehrilch import MoleculeSurface, Point, find_inside_cone_points


class Visualize:
    def __init__(
            self,
            surface1: MoleculeSurface,
            surface2: MoleculeSurface,
            point1: Point,
            point2: Point,
            area: float
    ):
        area1 = surface1.sphere_area(area)
        area2 = surface2.sphere_area(area)

        fi1 = math.degrees(math.acos(1 - area1 / (2 * math.pi * surface1.sphere_radius ** 2)))
        fi2 = math.degrees(math.acos(1 - area2 / (2 * math.pi * surface2.sphere_radius ** 2)))

        self.inside_points_idxes_1 = find_inside_cone_points(surface1.points, point1.origin_coords, fi1)
        self.inside_points_idxes_2 = find_inside_cone_points(surface2.points, point2.origin_coords, fi2)

        self.norm1 = point1.compute_norm(surface1)
        self.norm2 = point1.compute_norm(surface2)

        self.beta1 = self.__get_beta(point1)
        self.beta2 = self.__get_beta(point2)

        self.gamma1 = self.__get_gamma(point1)
        self.gamma2 = self.__get_gamma(point2)

    def draw_region(self):
        pass

    def draw_align(self):
        pass

    @staticmethod
    def __get_beta(point: Point):
        a = (point.origin_coords[0] ** 2 + point.origin_coords[1] ** 2) ** 0.5
        return math.asin(a / np.linalg.norm(point.origin_coords))

    @staticmethod
    def __get_gamma(point: Point):
        if point.origin_coords[1] == 0: return math.pi / 2  # todo: check
        return math.atan(point.origin_coords[0] / point.origin_coords[1])
