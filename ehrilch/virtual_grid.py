import math
import numpy as np

from ehrilch import MoleculeSurface, get_rotated_vector


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
        self.envs = [[self.last_point]]
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
        # self.transition_vector = self.get_transition_vector()
        # self.last_point = self.norm + self.transition_vector

        new_norm = np.copy(self.center_point) * math.cos(new_alpha)
        new_step_len = self.surface.sphere_radius * math.sin(new_alpha)
        base_vector = np.array((new_step_len, 0, 0), dtype='float64')

        rotation_angle = 60 / self.env_vect_count

        for rotation_idx in range(self.env_vect_count * 6):
            vect = get_rotated_vector(base_vector, rotation_angle * rotation_idx + self.delta_alpha, 'z')
            vect = get_rotated_vector(vect, self.beta, 'x')
            vect = get_rotated_vector(vect, self.gamma, 'z')
            new_points.append(vect + new_norm)

        self.envs.append(new_points)

    def blosum_score(self, other_grid):
        pass

    def geometry_score(self, other_grid):
        pass

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
