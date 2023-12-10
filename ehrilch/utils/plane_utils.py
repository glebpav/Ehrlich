import numpy as np
import math


from ehrilch.utils.math_utils import double_range


class GirdPoint(object):
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.connections = []
        self.idx = None
        self.localeIdx = None
        self.distance = None
        self.connectionsOnItsLevel = 0

    def __str__(self):
        print(self.x, self.y, self.z, self.radius)

    def add_connection(self, connection):
        if connection not in self.connections:
            self.connections.append(connection)

    def get_count_of_connections(self):
        return len(self.connections)

    def get_count_of_connections_on_its_level(self):
        return self.connectionsOnItsLevel

    def distance_to_point(self, point):
        dx = self.x - point.x
        dy = self.y - point.y
        dz = self.z - point.z
        self.distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    def __lt__(self, other):
        return self.distance < other.distance


class Connection(object):
    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node


class Frame(object):

    def __init__(self, length, width, grid_step, min_x, min_y, count_of_points_for_circle, error, water_radius):
        self.length = length
        self.width = width
        self.min_x = round(min_x, 3)
        self.min_y = round(min_y, 3)
        self.grid_step = grid_step
        self.count_of_points_for_circle = count_of_points_for_circle
        self.error = error
        self.water_radius = water_radius
        self.max_chain_distance = grid_step * math.sqrt(2) + 0.003
        self.grid = np.zeros((round(length / grid_step), round(width / grid_step)), int)
        self.ruler_x = [self.min_x + step * grid_step for step in range(0, round(length / grid_step))]
        self.ruler_y = [self.min_y + step * grid_step for step in range(0, round(width / grid_step))]
        self.atomsInFrame = []
        self.min_x_for_atom = None
        self.max_x_for_atom = None
        self.min_y_for_atom = None
        self.max_y_for_atom = None

    def get_grid_point(self, x, y, only_x=False):
        min_dx = abs(self.ruler_x[0] - x)
        min_didx = 0
        for idx in range(len(self.ruler_x)):
            dx = abs(self.ruler_x[idx] - x)
            if dx < min_dx:
                min_dx = dx
                min_didx = idx

        min_didy = 0
        if not only_x:
            min_dy = abs(self.ruler_y[0] - y)
            for idx in range(len(self.ruler_y)):
                dy = abs(self.ruler_y[idx] - y)
                if dy < min_dy:
                    min_dy = dy
                    min_didy = idx

        return min_didx, min_didy

    def process_circle(self, center_pos, radius):

        self.atomsInFrame.append(GirdPoint(center_pos[0], center_pos[1], None, radius))
        x, y = self.get_grid_point(center_pos[0], center_pos[1])
        self.grid[x, y] = 1

        for point in double_range(0, 2 * math.pi, (2 * math.pi) / self.count_of_points_for_circle / radius):
            x = radius * math.sin(point) + center_pos[0]
            y = radius * math.cos(point) + center_pos[1]

            x, y = self.get_grid_point(x, y)
            self.grid[x, y] = 1

    def is_point_in_circle(self, point_pos, circle_pos, radius, account_contre=False):
        dx = point_pos[0] - circle_pos[0]
        dy = point_pos[1] - circle_pos[1]
        if account_contre:
            if dx ** 2 + dy ** 2 < (radius + self.error) ** 2:
                return True
        else:
            if dx ** 2 + dy ** 2 < (radius - self.error) ** 2:
                return True
        return False

    def get_coordinates_by_idxs(self, idx_x, idx_y):
        if idx_x >= len(self.ruler_x) or idx_y >= len(self.ruler_y) or idx_x < 0 or idx_y < 0:
            return 0, 0
        return self.ruler_x[idx_x], self.ruler_y[idx_y]

    def is_molecule_contain_contur(self, molecule_pos, molecule_radius):

        begin_with_x, begin_with_y = self.get_grid_point(molecule_pos[0] - molecule_radius - self.error,
                                                         molecule_pos[1] - molecule_radius - self.error)
        end_with_x, end_with_y = self.get_grid_point(molecule_pos[0] + molecule_radius + self.error,
                                                     molecule_pos[1] + molecule_radius + self.error)

        for x_idx in range(begin_with_x, end_with_x):
            for y_idx in range(begin_with_y, end_with_y):
                if self.grid[x_idx, y_idx] == 1:
                    x, y = self.get_coordinates_by_idxs(x_idx, y_idx)
                    dx = molecule_pos[0] - x
                    dy = molecule_pos[1] - y
                    if dx ** 2 + dy ** 2 < (molecule_radius + self.error) ** 2:
                        return True

        return False

    def paint_frame_with_circle_reworked(self, circle_pos, radius):

        begin_with_x, begin_with_y = self.get_grid_point(circle_pos[0] - radius - self.error,
                                                         circle_pos[1] - radius - self.error)
        end_with_x, end_with_y = self.get_grid_point(circle_pos[0] + radius + self.error,
                                                     circle_pos[1] + radius + self.error)

        for x_idx in range(begin_with_x, end_with_x):
            for y_idx in range(begin_with_y, end_with_y):
                if self.grid[x_idx, y_idx] == 2:
                    continue
                x, y = self.get_coordinates_by_idxs(x_idx, y_idx)
                dx = circle_pos[0] - x
                dy = circle_pos[1] - y
                if self.is_point_in_circle((x, y), circle_pos, radius):
                    self.grid[x_idx, y_idx] = 2

    def paint_frame_with_circle_reworked2(self, circle_pos, radius):

        count_of_columns = self.get_grid_point(0, circle_pos[1] + radius)[1] - self.get_grid_point(0, circle_pos[1])[1]

        max_value = radius
        start_x = self.get_grid_point(circle_pos[0] - max_value, 0, only_x=True)[0]
        end_x, y = self.get_grid_point(circle_pos[0] + max_value, circle_pos[1])

        self.grid[start_x: end_x, y] = 2

        for i in range(1, count_of_columns):
            max_value = math.cos(math.pi * i / count_of_columns) * radius
            start_x = self.get_grid_point(circle_pos[0] - max_value, 0, only_x=True)[0]
            end_x = self.get_grid_point(circle_pos[0] + max_value, 0, only_x=True)[0]

            self.grid[start_x: end_x, y + i] = 2
            self.grid[start_x: end_x, y - i] = 2

    def clear_contur(self):
        for x_idx in range(len(self.grid)):
            for y_idx in range(len(self.grid[x_idx])):
                if self.grid[x_idx, y_idx] == 1:
                    self.grid[x_idx, y_idx] = 0

    def count_water_neighbors(self, x_idx, y_idx):
        neighbors_count = 0

        max_x = len(self.grid)
        max_y = len(self.grid[0])

        if x_idx - 1 >= 0 and y_idx - 1 >= 0:
            if self.grid[x_idx - 1, y_idx - 1] == 2:
                neighbors_count += 1
        if y_idx - 1 >= 0:
            if self.grid[x_idx, y_idx - 1] == 2:
                neighbors_count += 1
        if x_idx + 1 < max_x and y_idx - 1 >= 0:
            if self.grid[x_idx + 1, y_idx - 1] == 2:
                neighbors_count += 1

        if x_idx - 1 >= 0:
            if self.grid[x_idx - 1, y_idx] == 2:
                neighbors_count += 1
        if x_idx + 1 < max_x:
            if self.grid[x_idx + 1, y_idx] == 2:
                neighbors_count += 1

        if x_idx - 1 >= 0 and y_idx + 1 < max_y:
            if self.grid[x_idx - 1, y_idx + 1] == 2:
                neighbors_count += 1
        if y_idx + 1 < max_y:
            if self.grid[x_idx, y_idx + 1] == 2:
                neighbors_count += 1
        if x_idx + 1 < max_x and y_idx + 1 < max_y:
            if self.grid[x_idx + 1, y_idx + 1] == 2:
                neighbors_count += 1

        return neighbors_count

    def count_marked_point(self, x_idx, y_idx, mark):
        neighbors_count = 0

        max_x = len(self.grid)
        max_y = len(self.grid[0])

        if x_idx - 1 >= 0 and y_idx - 1 >= 0:
            if self.grid[x_idx - 1, y_idx - 1] == mark:
                neighbors_count += 1
        if y_idx - 1 >= 0:
            if self.grid[x_idx, y_idx - 1] == mark:
                neighbors_count += 1
        if x_idx + 1 < max_x and y_idx - 1 >= 0:
            if self.grid[x_idx + 1, y_idx - 1] == mark:
                neighbors_count += 1

        if x_idx - 1 >= 0:
            if self.grid[x_idx - 1, y_idx] == mark:
                neighbors_count += 1
        if x_idx + 1 < max_x:
            if self.grid[x_idx + 1, y_idx] == mark:
                neighbors_count += 1

        if x_idx - 1 >= 0 and y_idx + 1 < max_y:
            if self.grid[x_idx - 1, y_idx + 1] == mark:
                neighbors_count += 1
        if y_idx + 1 < max_y:
            if self.grid[x_idx, y_idx + 1] == mark:
                neighbors_count += 1
        if x_idx + 1 < max_x and y_idx + 1 < max_y:
            if self.grid[x_idx + 1, y_idx + 1] == mark:
                neighbors_count += 1

        return neighbors_count

    def has_atoms_in_frame(self):
        return len(self.atomsInFrame) > 0

    def find_final_contur(self):
        for x_idx in range(len(self.grid)):
            for y_idx in range(len(self.grid[x_idx])):
                if self.grid[x_idx, y_idx] == 0:
                    neighbors = self.count_water_neighbors(x_idx, y_idx)
                    if 1 < neighbors:
                        self.grid[x_idx, y_idx] = 3

    def find_water_offset(self):
        for offset_contur in range(math.ceil(self.water_radius / self.grid_step)):
            for x_idx in range(len(self.grid)):
                for y_idx in range(len(self.grid[x_idx])):
                    if self.grid[x_idx, y_idx] == 0:
                        neighbors = self.count_marked_point(x_idx, y_idx, offset_contur + 3)
                        if 1 < neighbors:
                            self.grid[x_idx, y_idx] = offset_contur + 4

        for x_idx in range(len(self.grid)):
            for y_idx in range(len(self.grid[x_idx])):
                if self.grid[x_idx, y_idx] > 3:
                    self.grid[x_idx, y_idx] = 4

    def paint_frames(self):
        if len(self.atomsInFrame) == 0:
            self.grid[:, :] = 2
            return

        first_atom = self.atomsInFrame[0]
        self.min_x_for_atom = first_atom.x - first_atom.radius
        self.max_x_for_atom = first_atom.x + first_atom.radius
        self.min_y_for_atom = first_atom.y - first_atom.radius
        self.max_y_for_atom = first_atom.y + first_atom.radius

        for atom in self.atomsInFrame:
            submin_x_for_atom = atom.x - atom.radius
            submax_x_for_atom = atom.x + atom.radius
            submin_y_for_atom = atom.y - atom.radius
            submax_y_for_atom = atom.y + atom.radius

            if submin_x_for_atom < self.min_x_for_atom:
                self.min_x_for_atom = submin_x_for_atom

            if submin_y_for_atom < self.min_y_for_atom:
                self.min_y_for_atom = submin_y_for_atom

            if submax_x_for_atom > self.max_x_for_atom:
                self.max_x_for_atom = submax_x_for_atom

            if submax_y_for_atom > self.max_y_for_atom:
                self.max_y_for_atom = submax_y_for_atom

        self.min_x_for_atom, self.min_y_for_atom = self.get_grid_point(self.min_x_for_atom, self.min_y_for_atom)
        self.max_x_for_atom, self.max_y_for_atom = self.get_grid_point(self.max_x_for_atom, self.max_y_for_atom)

        if len(self.grid[0]) > self.max_y_for_atom + 1:
            self.max_y_for_atom += 1
        if len(self.grid) > self.max_x_for_atom + 1:
            self.max_x_for_atom += 1

        self.grid[:, 0:self.min_y_for_atom] = 2
        self.grid[:, self.max_y_for_atom:-1] = 2

        self.grid[0:self.min_x_for_atom, :] = 2
        self.grid[self.max_x_for_atom:-1, :] = 2
