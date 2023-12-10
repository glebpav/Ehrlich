import math
import numpy as np
from Bio.SubsMat import MatrixInfo

from ehrilch import MoleculeSurface, get_rotated_vector, get_dist, get_norm, get_cos


class VirtualPoint:
    def __init__(self, coords, sphere_point):
        self.coords = coords
        self.sphere_point = sphere_point


class VirtualGrid:

    def __init__(self, center_point, surface: MoleculeSurface, env_density=5):
        self.envs = None
        self.alpha = None
        self.env_vect_count = None
        self.was_adding = None
        self.gamma = None
        self.beta = None
        self.last_point = None

        self.center_point = center_point
        self.surface: MoleculeSurface = surface
        self.env_density = env_density
        self.edge = self.get_edge_len()
        self.angle = None
        self.delta_alpha = 0

    def build(self, angle=0.):
        self.edge = self.get_edge_len()
        self.delta_alpha = angle

        self.last_point = np.copy(self.center_point)
        self.beta = math.degrees(self.get_beta())
        self.gamma = math.degrees(self.get_gamma())
        self.was_adding = True
        self.env_vect_count = 1
        self.alpha = self.get_alpha()
        self.env_vect_count = 0
        self.envs = [[VirtualPoint(self.last_point, self.get_closes_point(self.last_point))]]
        self.add_env()

    def add_env(self):
        new_alpha = self.alpha * len(self.envs)

        if new_alpha > math.pi / 2:
            self.was_adding = False

        if self.was_adding:
            self.env_vect_count += 1
        else:
            self.env_vect_count -= 1
            if self.env_vect_count == 0:
                return
        new_points = []

        new_norm = np.copy(self.center_point) * math.cos(new_alpha)
        new_step_len = self.surface.sphere_radius * math.sin(new_alpha)
        base_vector = np.array((new_step_len, 0, 0), dtype='float64')

        rotation_angle = 60 / self.env_vect_count

        for rotation_idx in range(self.env_vect_count * 6):
            vect = get_rotated_vector(base_vector, rotation_angle * rotation_idx + self.delta_alpha, 'z')
            vect = get_rotated_vector(vect, self.beta, 'x')
            vect = get_rotated_vector(vect, self.gamma, 'z')
            new_points.append(VirtualPoint(vect + new_norm, self.get_closes_point(vect + new_norm)))

        self.envs.append(new_points)

    def get_closes_point(self, coords):
        dists = {idx: get_dist(coords, point.origin_coords) for idx, point in enumerate(self.surface.points)}
        dists = {k: v for k, v in sorted(dists.items(), key=lambda item: item[1])}
        best_point_idx = list(dists.keys())[0]
        return self.surface.points[best_point_idx]

    def blosum_score(self, other_grid):
        score = 0
        min_env_len = min(len(self.envs), len(other_grid.envs))

        for env_level_idx in range(min_env_len):
            min_env_level_len = min(len(self.envs[env_level_idx]), len(other_grid.envs[env_level_idx]))
            for point_idx in range(min_env_level_len):
                atom_idx_1 = self.envs[env_level_idx][point_idx].sphere_point.atom_idx
                acid1 = self.surface.molecule.atoms[atom_idx_1].residue[0]

                atom_idx_2 = other_grid.envs[env_level_idx][point_idx].sphere_point.atom_idx
                acid2 = other_grid.surface.molecule.atoms[atom_idx_2].residue[0]

                try:
                    score += MatrixInfo.blosum50[(acid1, acid2)]
                except:
                    score += MatrixInfo.blosum50[(acid2, acid1)]
        return score

    def get_geometry_params(self):
        dists = []
        angles = []
        pass

        center_coords = np.array([0., 0., 0.])
        point = self.envs[0][0].sphere_point
        center_vector = point.shrinked_coords

        if len(self.envs) < 2:
            return

        norm_components = []
        for idx in range(1, len(self.envs[1])):
            vect1 = self.envs[1][idx - 1].origin_coords - point.origin_coords
            vect2 = self.envs[1][idx].origin_coords - point.origin_coords
            res_vect = np.cross(vect1, vect2)
            norm_components.append(get_norm(res_vect))
        avg_adj_vect = np.average(np.array(norm_components), axis=0)
        is_norm_inverted = get_cos(avg_adj_vect, point.origin_coords) > 0
        # print(f"cos is: {get_cos(avg_adj_vect, point.origin_coords)}")

        norm_components = []
        for i in range(1, 3):
            for idx in range(1, len(self.envs[1])):
                vect1 = self.envs[i][idx - 1].shrinked_coords - center_vector
                vect2 = self.envs[i][idx].shrinked_coords - center_vector
                res_vect = np.cross(vect1, vect2)
                norm_components.append(get_norm(res_vect))

        avg_adj_vect = np.average(np.array(norm_components), axis=0)
        norm_avg = np.array(get_norm(avg_adj_vect))

        if is_norm_inverted:
            norm_avg *= -1

        for env_level in self.envs:
            dist_level = []
            angle_level = []
            for env_point in env_level:
                vect = env_point.shrinked_coords - point.shrinked_coords
                dist_level.append(get_dist(point.shrinked_coords, env_point.shrinked_coords))
                angle_level.append(math.acos(get_cos(norm_avg, vect)))
            dists.append(dist_level)
            angles.append(angle_level)

        return norm_avg, dists, angles

    def geometry_score(self, other_grid, threshold, geometry_effect):

        score = 0

        norm1_avg, dists1, angles1 = self.get_geometry_params()
        norm2_avg, dists2, angles2 = other_grid.get_geometry_params()

        height_difference = []
        for level_idx in range(1, len(dists1)):
            level_dif = []
            for idx in range(len(dists1[level_idx])):
                a = dists1[level_idx][idx] ** 2 + dists2[level_idx][idx] ** 2 - 2 * dists1[level_idx][idx] * \
                    dists2[level_idx][idx] * math.cos(angles1[level_idx][idx] - angles2[level_idx][idx])
                level_dif.append(math.sqrt(a))
            height_difference.append(level_dif)
        # height_difference

        min_env_len = min(len(self.envs), len(other_grid.envs))

        for env_level_idx in range(min_env_len):
            min_env_level_len = min(len(self.envs[env_level_idx]), len(other_grid.envs[env_level_idx]))
            for point_idx in range(min_env_level_len):

                c = angles1[env_level_idx][point_idx] - angles2[env_level_idx][point_idx]
                b = 2 * dists1[env_level_idx][point_idx] * dists2[env_level_idx][point_idx] * math.cos(c)
                a = dists1[env_level_idx][point_idx] ** 2 + dists2[env_level_idx][point_idx] ** 2 - b
                geometry_dif = math.sqrt(a)

                geometry_impact = 0
                if geometry_dif > threshold:
                    geometry_impact = geometry_effect * geometry_dif

                score += geometry_impact

        return score

    # first basis rotation angle
    def get_beta(self):
        a = (self.center_point[0] ** 2 + self.center_point[1] ** 2) ** 0.5
        return math.asin(a / np.linalg.norm(self.center_point))

    # second basis rotation angle
    def get_gamma(self):
        if self.center_point[1] == 0: return math.pi / 2  # todo: check
        return math.atan(self.center_point[0] / self.center_point[1])

    # next env ration angle
    def get_alpha(self):
        alpha = math.acos(
            1 - (float(self.edge * self.env_vect_count) ** 2 / (2 * float(self.surface.sphere_radius) ** 2)))

        if not self.was_adding:
            alpha = math.pi - alpha
        else:
            alpha = alpha

        if 1 - (float(self.edge * self.env_vect_count + 1) ** 2 / (2 * float(self.surface.sphere_radius) ** 2)) < -1:
            self.was_adding = False
        return alpha

    def get_edge_len(self):
        base_area = 10000
        fi = math.degrees(math.acos(1 - base_area / (2 * math.pi * self.surface.sphere_radius ** 2)))
        return (2 * math.pi * self.surface.sphere_radius * fi) / (360 * self.env_density)
