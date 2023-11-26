import matplotlib.pyplot as plt
import numpy as np
import decimal
import math
import pickle

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal
from decimal import *

# radius

water_radius = 1.4
atom_radius = 2.1

# count of connections for each point
connections_amount = 4

#
count_of_points_for_circle = 300
step_water_molecule = 1

# permissible error
error = 0.009

# by default x and z steps must be the same
# because connections between points won't work correct
z_step = 2.0
x_step = 2.0

MAX_VWR = 2.1

lower_step_x = 1.4


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


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

    def __init__(self, length, width, grid_step, min_x, min_y):
        self.length = length
        self.width = width
        self.min_x = round(min_x, 3)
        self.min_y = round(min_y, 3)
        self.grid_step = grid_step
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

        for point in drange(0, 2 * math.pi, (2 * math.pi) / count_of_points_for_circle / radius):
            x = radius * math.sin(point) + center_pos[0]
            y = radius * math.cos(point) + center_pos[1]

            x, y = self.get_grid_point(x, y)
            self.grid[x, y] = 1

    def is_point_in_circle(self, point_pos, circle_pos, radius, account_contre=False):
        dx = point_pos[0] - circle_pos[0]
        dy = point_pos[1] - circle_pos[1]
        if account_contre:
            if dx ** 2 + dy ** 2 < (radius + error) ** 2:
                return True
        else:
            if dx ** 2 + dy ** 2 < (radius - error) ** 2:
                return True
        return False

    def get_coordinates_by_idxs(self, idx_x, idx_y):
        if idx_x >= len(self.ruler_x) or idx_y >= len(self.ruler_y) or idx_x < 0 or idx_y < 0:
            return 0, 0
        return self.ruler_x[idx_x], self.ruler_y[idx_y]

    def is_molecule_contain_contur(self, molecule_pos, molecule_radius):

        begin_with_x, begin_with_y = self.get_grid_point(molecule_pos[0] - molecule_radius - error,
                                                         molecule_pos[1] - molecule_radius - error)
        end_with_x, end_with_y = self.get_grid_point(molecule_pos[0] + molecule_radius + error,
                                                     molecule_pos[1] + molecule_radius + error)

        for x_idx in range(begin_with_x, end_with_x):
            for y_idx in range(begin_with_y, end_with_y):
                if self.grid[x_idx, y_idx] == 1:
                    x, y = self.get_coordinates_by_idxs(x_idx, y_idx)
                    dx = molecule_pos[0] - x
                    dy = molecule_pos[1] - y
                    if dx ** 2 + dy ** 2 < (molecule_radius + error) ** 2:
                        return True

        return False

    def paint_frame_with_circle_reworked(self, circle_pos, radius):

        begin_with_x, begin_with_y = self.get_grid_point(circle_pos[0] - radius - error, circle_pos[1] - radius - error)
        end_with_x, end_with_y = self.get_grid_point(circle_pos[0] + radius + error, circle_pos[1] + radius + error)

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

    def countWaterNeighbors(self, x_idx, y_idx):
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
                    neighbors = self.countWaterNeighbors(x_idx, y_idx)
                    if 1 < neighbors:
                        self.grid[x_idx, y_idx] = 3

    def find_water_offset(self):
        for offset_contur in range(math.ceil(water_radius / self.grid_step)):
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


def delete_unimportant(coords):
    coords = np.array([[atom[0], atom[1], atom[2]] for atom in coords])
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    max_radius = MAX_VWR

    max_z = max(z) + max_radius + water_radius * 2
    min_z = min(z) - max_radius - water_radius * 2

    min_x = min(x) - max_radius - water_radius * 2
    max_x = max(x) + max_radius + water_radius * 2

    min_y = min(y) - max_radius - water_radius * 2
    max_y = max(y) + max_radius + water_radius * 2

    print(max_z, min_z, max_radius)

    points = [GirdPoint(float(coord[0]), float(coord[1]), float(coord[2]), atom_radius) for coord in coords]

    for idx, point in enumerate(points):
        point.idx = idx

    levels = []
    for currentZ in drange(Decimal(min_z), Decimal(max_z), z_step):

        current_frame = Frame(max_x - min_x + error, max_y - min_y + error, lower_step_x, min_x, min_y)

        print(float('{:.2f}'.format(currentZ)), 'out of', float('{:.2f}'.format(max_z)))

        for point in points:
            if (point.z > currentZ > (point.z - point.radius)) or (point.z < currentZ < point.z + point.radius):
                new_radius = ((point.radius ** 2) - ((currentZ - point.z) ** 2)) ** 0.5
                current_frame.process_circle((point.x, point.y), new_radius)

        levels.append(current_frame)

    count_of_levels = len(levels)
    for idx, frame in enumerate(levels):

        frame.paint_frames()
        cells_for_water = round(water_radius / frame.grid_step)

        print(idx, "out of:", count_of_levels)

        if frame.has_atoms_in_frame():
            for x_idx in range(frame.min_x_for_atom - cells_for_water - 1, frame.max_x_for_atom + cells_for_water + 1,
                               step_water_molecule):

                for y_idx in range(frame.min_y_for_atom - cells_for_water - 1,
                                   frame.max_y_for_atom + cells_for_water + 1,
                                   step_water_molecule):
                    x, y = frame.get_coordinates_by_idxs(x_idx, y_idx)
                    if not frame.is_molecule_contain_contur((x, y), water_radius):
                        frame.paint_frame_with_circle_reworked((x, y), water_radius)

        frame.clear_contur()
        frame.find_final_contur()
        for i in range(math.ceil(water_radius / frame.grid_step)):
            frame.find_water_offset()

    protein_atoms = coords.copy()
    out_atoms = []
    saved_idxs = []
    z_scale = (max_z - min_z) / len(levels)

    for atom_idx, atom in enumerate(protein_atoms):
        z_idx = math.floor((atom[2] - min_z) / z_scale)
        frame = levels[z_idx]
        x_idx, y_idx = frame.get_grid_point(atom[0], atom[1])
        if frame.grid[x_idx, y_idx] != 0:
            out_atoms.append(atom)
            saved_idxs.append(atom_idx)

    out_atoms = np.array(out_atoms)

    print(f"Before optimizing: {len(coords)}")
    print(f"After optimizing: {len(out_atoms)}")
    print(f"Optimizing ratio: {1 - len(out_atoms) / len(coords)}")

    return saved_idxs
