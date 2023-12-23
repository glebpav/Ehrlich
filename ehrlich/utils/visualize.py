import math

import numpy as np
from matplotlib import pyplot as plt

from ehrlich import MoleculeSurface, Point, find_inside_cone_points, get_rotated_vector


class Visualize:
    def __init__(
            self,
            surface1: MoleculeSurface,
            surface2: MoleculeSurface,
            point1: Point,
            point2: Point,
            area: float
    ):

        self.surface1 = surface1
        self.surface2 = surface2

        area1 = surface1.sphere_area(area)
        area2 = surface2.sphere_area(area)

        fi1 = math.degrees(math.acos(1 - area1 / (2 * math.pi * surface1.sphere_radius ** 2)))
        fi2 = math.degrees(math.acos(1 - area2 / (2 * math.pi * surface2.sphere_radius ** 2)))

        self.inside_points_idxes_1 = find_inside_cone_points(surface1.points, point1.origin_coords, fi1)
        self.inside_points_idxes_2 = find_inside_cone_points(surface2.points, point2.origin_coords, fi2)

        self.norm1 = point1.compute_norm(surface1)
        self.norm2 = point2.compute_norm(surface2)

        self.offset_1 = point1.shrunk_coords
        self.offset_2 = point2.shrunk_coords

        self.beta1 = math.degrees(self.__get_beta(self.norm1))
        self.beta2 = math.degrees(self.__get_beta(self.norm2))

        self.gamma1 = math.degrees(self.__get_gamma(self.norm1))
        self.gamma2 = math.degrees(self.__get_gamma(self.norm2))

    def draw_region(self, elevation=30, azimuth=45, opacity=0.7, fig_size=(20, 10)):
        figure, axis = plt.subplots(1, 2, figsize=fig_size, subplot_kw=dict(projection='3d'), dpi=500)
        X1, Y1, Z1 = [], [], []
        Xa1, Ya1, Za1 = [], [], []
        X2, Y2, Z2 = [], [], []
        Xa2, Ya2, Za2 = [], [], []

        converted_inside_cone_points_1 = [
            np.array(self.__convert_coord(point.shrunk_coords - self.offset_1, self.beta1, self.gamma1))
            for point_idx, point in enumerate(self.surface1.points)
            if point_idx in self.inside_points_idxes_1
        ]

        converted_inside_cone_atoms_1 = [
            np.array(self.__convert_coord(
                self.surface1.molecule.atoms[point.atom_idx].coords - self.offset_1,
                self.beta1, self.gamma1
            ))
            for point_idx, point in enumerate(self.surface1.points)
            if point_idx in self.inside_points_idxes_1
        ]

        converted_inside_cone_points_2 = [
            np.array(self.__convert_coord(point.shrunk_coords - self.offset_2, self.beta2, self.gamma2))
            for point_idx, point in enumerate(self.surface2.points)
            if point_idx in self.inside_points_idxes_2
        ]

        converted_inside_cone_atoms_2 = [
            np.array(self.__convert_coord(
                self.surface2.molecule.atoms[point.atom_idx].coords - self.offset_2,
                self.beta2, self.gamma2
            ))
            for point_idx, point in enumerate(self.surface2.points)
            if point_idx in self.inside_points_idxes_2
        ]

        for point in converted_inside_cone_points_1:
            X1.append(point[0])
            Y1.append(point[1])
            Z1.append(point[2])

        for point in converted_inside_cone_atoms_1:
            Xa1.append(point[0])
            Ya1.append(point[1])
            Za1.append(point[2])

        for point in converted_inside_cone_points_2:
            X2.append(point[0])
            Y2.append(point[1])
            Z2.append(point[2])

        for point in converted_inside_cone_atoms_2:
            Xa2.append(point[0])
            Ya2.append(point[1])
            Za2.append(point[2])

        X1 = np.array(X1)
        Y1 = np.array(Y1)
        Z1 = np.array(Z1)

        X2 = np.array(X2)
        Y2 = np.array(Y2)
        Z2 = np.array(Z2)

        axis[0].view_init(elev=elevation, azim=azimuth, roll=0)
        axis[1].view_init(elev=elevation, azim=azimuth, roll=0)

        self.norm1 *= 5
        self.norm2 *= 5

        print(f"origin norm1: {self.norm1}")
        print(f"origin norm2: {self.norm2}")

        norm1 = self.__convert_coord(self.norm1, self.beta1, self.gamma1)
        norm2 = self.__convert_coord(self.norm2, self.beta2, self.gamma2)

        print(f"converted norm1: {norm1}")
        print(f"converted norm2: {norm2}")

        axis[0].quiver(0, 0, 0, norm1[0], norm1[1], norm1[2], color="red")
        axis[1].quiver(0, 0, 0, norm2[0], norm2[1], norm2[2], color="blue")

        axis[0].scatter(X1, Y1, Z1, color=["red"] * len(X1), alpha=opacity)
        axis[0].scatter(Xa1, Ya1, Za1, color=["black"] * len(X1), alpha=1)
        axis[1].scatter(X2, Y2, Z2, color=["blue"] * len(X2), alpha=opacity)
        axis[1].scatter(Xa2, Ya2, Za2, color=["black"] * len(X2), alpha=1)

        plt.savefig('fig2.png')
        return figure

    def draw_align(self, elevation=30, azimuth=45, opacity=0.7, fig_size=(10, 10)):
        figure, axis = plt.subplots(1, 1, figsize=fig_size, subplot_kw=dict(projection='3d'), dpi=500)
        X1, Y1, Z1 = [], [], []
        X2, Y2, Z2 = [], [], []

        converted_inside_cone_points_1 = [
            np.array(self.__convert_coord(point.shrunk_coords - self.offset_1, self.beta1, self.gamma1))
            for point_idx, point in enumerate(self.surface1.points)
            if point_idx in self.inside_points_idxes_1
        ]

        converted_inside_cone_points_2 = [
            np.array(self.__convert_coord(point.shrunk_coords - self.offset_2, self.beta2, self.gamma2))
            for point_idx, point in enumerate(self.surface2.points)
            if point_idx in self.inside_points_idxes_2
        ]

        for point in converted_inside_cone_points_1:
            X1.append(point[0])
            Y1.append(point[1])
            Z1.append(point[2])

        for point in converted_inside_cone_points_2:
            X2.append(point[0])
            Y2.append(point[1])
            Z2.append(point[2])

        X1 = np.array(X1)
        Y1 = np.array(Y1)
        Z1 = np.array(Z1)

        X2 = np.array(X2)
        Y2 = np.array(Y2)
        Z2 = np.array(Z2)

        axis.view_init(elev=elevation, azim=azimuth, roll=0)

        self.norm1 *= 5
        self.norm2 *= 5

        norm1 = self.__convert_coord(self.norm1, self.beta1, self.gamma1)
        norm2 = self.__convert_coord(self.norm2, self.beta2, self.gamma2)

        axis.quiver(0, 0, 0, norm1[0], norm1[1], norm1[2], color="red")
        axis.quiver(0, 0, 0, norm2[0], norm2[1], norm2[2], color="blue")

        axis.scatter(X1, Y1, Z1, color=["red"] * len(X1), alpha=opacity)
        axis.scatter(X2, Y2, Z2, color=["blue"] * len(X2), alpha=opacity)

        plt.savefig('fig1.png')
        return figure

    @staticmethod
    def __get_beta(point):
        a = (point[0] ** 2 + point[1] ** 2) ** 0.5
        if point[2] < 0:
            return math.pi - math.asin(round(a / np.linalg.norm(point), 5))
        else:
            return math.asin(round(a / np.linalg.norm(point), 5))

    @staticmethod
    def __get_gamma(point):
        if (point[1] * point[2] > 0 and (point[1] < 0 or point[0] < 0)) or (point[1] < 0 < point[2]):
            return math.pi + math.atan(point[0] / point[1])
        if (point[0] < 0 and point[1] > 0):
            return math.fabs(math.atan(point[0] / point[1]))
        if point[1] == 0:
                return math.pi / 2  # todo: check
        return math.atan(point[0] / point[1])

    def __convert_coord(self, point, beta, gamma):
        point1 = get_rotated_vector(point, -gamma, 'z')
        point1 = get_rotated_vector(point1, -beta, 'x')
        return point1
