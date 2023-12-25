import math

import numpy as np
from matplotlib import pyplot as plt

from ehrlich import MoleculeSurface, Point, find_inside_cone_points, get_rotated_vector

ATOM_VWR = {
    'C': 1.9080,
    'H': 1.387,
    'N': 1.8240,
    'S': 2.0000,
    'P': 2.1000,
    'O': 1.6612,
    'N1+': 1.8240
}


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

        # print(f"original norm 1: {self.norm1}")
        # print(f"converted norm 1: {self.__convert_coord(self.norm1, self.beta1, self.gamma1)}")
        # print()
        # print(f"original norm 2: {self.norm2}")
        # print(f"converted norm 2: {self.__convert_coord(self.norm2, self.beta2, self.gamma2)}")

    def draw_region(self, elevation=30, azimuth=45, opacity=0.7, fig_size=(20, 10), dpi=300):
        figure, axis = plt.subplots(1, 2, figsize=fig_size, subplot_kw=dict(projection='3d'), dpi=dpi)

        X1, Y1, Z1 = [], [], []
        Xa1, Ya1, Za1 = [], [], []
        X2, Y2, Z2 = [], [], []
        Xa2, Ya2, Za2 = [], [], []

        residue_numbers_1 = set([
            self.surface1.molecule.atoms[point.atom_idx].residue_num
            for point_idx, point in enumerate(self.surface1.points)
            if point_idx in self.inside_points_idxes_1
        ])

        residue_numbers_2 = set([
            self.surface2.molecule.atoms[point.atom_idx].residue_num
            for point_idx, point in enumerate(self.surface2.points)
            if point_idx in self.inside_points_idxes_2
        ])

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
            if self.surface1.molecule.atoms[point.atom_idx].residue_num in residue_numbers_1
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
            if self.surface2.molecule.atoms[point.atom_idx].residue_num in residue_numbers_2
        ]

        radius_1 = [
            ATOM_VWR[self.surface1.molecule.atoms[point.atom_idx].name] * 70
            for point_idx, point in enumerate(self.surface1.points)
            if self.surface1.molecule.atoms[point.atom_idx].residue_num in residue_numbers_1
        ]

        radius_2 = [
            ATOM_VWR[self.surface2.molecule.atoms[point.atom_idx].name] * 70
            for point_idx, point in enumerate(self.surface2.points)
            if self.surface2.molecule.atoms[point.atom_idx].residue_num in residue_numbers_2
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

        max1 = max(np.abs(np.max(X1)), np.abs(np.max(Y1)), np.abs(np.max(Z1)))
        max2 = max(np.abs(np.max(X2)), np.abs(np.max(Y2)), np.abs(np.max(Z2)))

        axis[0].set_xlim([-max1, max1])
        axis[0].set_ylim([-max1, max1])
        axis[0].set_zlim([-max1, max1])

        axis[1].set_xlim([-max2, max2])
        axis[1].set_ylim([-max2, max2])
        axis[1].set_zlim([-max2, max2])

        axis[0].view_init(elev=elevation, azim=azimuth, roll=0)
        axis[1].view_init(elev=elevation, azim=azimuth, roll=0)

        self.norm1 *= 5
        self.norm2 *= 5

        norm1 = self.__convert_coord(self.norm1, self.beta1, self.gamma1)
        norm2 = self.__convert_coord(self.norm2, self.beta2, self.gamma2)

        axis[0].quiver(0, 0, 0, norm1[0], norm1[1], norm1[2], color="red")
        axis[1].quiver(0, 0, 0, norm2[0], norm2[1], norm2[2], color="blue")

        axis[0].scatter(Xa1, Ya1, Za1, color=["#717171"] * len(Xa1), alpha=1,
                        s=[(radius ** 2 * math.pi / 4) / (2 * max1) * (1 / 3) * fig_size[0] for radius in radius_1])
        axis[0].scatter(X1, Y1, Z1, color=["red"] * len(X1), alpha=opacity)
        axis[1].scatter(Xa2, Ya2, Za2, color=["#717171"] * len(Xa2), alpha=1,
                        s=[(radius ** 2 * math.pi / 4) / (2 * max2) * (1 / 3) * fig_size[0] for radius in radius_2])
        axis[1].scatter(X2, Y2, Z2, color=["blue"] * len(X2), alpha=opacity)

        for point_idx in self.inside_points_idxes_1:
            for adj_point_idx in self.surface1.points[point_idx].neighbors_points_idx:

                if adj_point_idx not in self.inside_points_idxes_1:
                    continue

                iv = self.surface1.points[point_idx].shrunk_coords
                jv = self.surface1.points[adj_point_idx].shrunk_coords

                iv = self.__convert_coord(iv - self.offset_1, self.beta1, self.gamma1)
                jv = self.__convert_coord(jv - self.offset_1, self.beta1, self.gamma1)

                axis[0].plot3D(*[a for a in zip(iv, jv)], color='#8E0EFD', alpha=opacity)

        for point_idx in self.inside_points_idxes_2:
            for adj_point_idx in self.surface2.points[point_idx].neighbors_points_idx:

                if adj_point_idx not in self.inside_points_idxes_2:
                    continue

                iv = self.surface2.points[point_idx].shrunk_coords
                jv = self.surface2.points[adj_point_idx].shrunk_coords

                iv = self.__convert_coord(iv - self.offset_2, self.beta2, self.gamma2)
                jv = self.__convert_coord(jv - self.offset_2, self.beta2, self.gamma2)

                axis[1].plot3D(*[a for a in zip(iv, jv)], color='#8E0EFD', alpha=opacity)

        return figure

    def draw_align(self, elevation=30, azimuth=45, opacity=0.7, fig_size=(10, 10), dpi=300):
        figure, axis = plt.subplots(1, 1, figsize=fig_size, subplot_kw=dict(projection='3d'), dpi=dpi)
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

        return figure

    @staticmethod
    def __get_beta(point):
        a = (point[0] ** 2 + point[1] ** 2) ** 0.5
        if point[2] < 0:
            return math.pi - math.asin(round(a / np.linalg.norm(point), 7))
        else:
            return math.asin(round(a / np.linalg.norm(point), 7))

    @staticmethod
    def __get_gamma(point):
        if (point[1] * point[0] > 0.00000 and (point[1] < 0.000 or point[0] < 0.000000)) or (
                point[1] < 0.0000 and 0. < point[0]):
            # print("here1")
            return math.pi + math.atan(point[0] / point[1])
        if point[0] < 0.000 and point[1] > 0.000:
            # print("here")
            # print(math.degrees(1. * math.pi - math.fabs(math.atan(point[0] / point[1]))))
            return - math.fabs(math.atan(point[0] / point[1]))
        if point[1] == 0:
            return math.pi / 2  # todo: check
        return math.atan(point[0] / point[1])

    def __convert_coord(self, point, beta, gamma):
        point1 = get_rotated_vector(point, -gamma, 'z')
        # print(f"after gamma rotation: {point1}")
        point1 = get_rotated_vector(point1, -beta, 'x')
        return point1
