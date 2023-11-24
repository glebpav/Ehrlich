import matplotlib.pyplot as plt
import numpy as np
import decimal
import math
import pickle

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal
from decimal import *

# radius es

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

    def addConnection(self, connection):
        if connection not in self.connections:
            self.connections.append(connection)

    def getCountOfConnections(self):
        return len(self.connections)

    def getCountOfConnectionsOnItsLevel(self):
        return self.connectionsOnItsLevel

    def distanceToPoint(self, point):
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

    def is_point_in_circle(self, pointPos, circlePos, radius, accountConture=False):
        dx = pointPos[0] - circlePos[0]
        dy = pointPos[1] - circlePos[1]
        if accountConture:
            if dx ** 2 + dy ** 2 < (radius + error) ** 2:
                return True
        else:
            if dx ** 2 + dy ** 2 < (radius - error) ** 2:
                return True
        return False

    def getCoordinatesByIdxs(self, idx_x, idx_y):
        if idx_x >= len(self.ruler_x) or idx_y >= len(self.ruler_y) or idx_x < 0 or idx_y < 0:
            return (0, 0)
        return (self.ruler_x[idx_x], self.ruler_y[idx_y])

    def isMoleculeContainContuer(self, moleculePos, moleculeRadius):

        beginWithX, beginWithY = self.get_grid_point(moleculePos[0] - moleculeRadius - error,
                                                     moleculePos[1] - moleculeRadius - error)
        endWithX, endWithY = self.get_grid_point(moleculePos[0] + moleculeRadius + error,
                                                 moleculePos[1] + moleculeRadius + error)

        for x_idx in range(beginWithX, endWithX):
            for y_idx in range(beginWithY, endWithY):
                if self.grid[x_idx, y_idx] == 1:
                    x, y = self.getCoordinatesByIdxs(x_idx, y_idx)
                    dx = moleculePos[0] - x
                    dy = moleculePos[1] - y
                    if dx ** 2 + dy ** 2 < (moleculeRadius + error) ** 2:
                        return True

        return False

    def paintFrameWithCircle(self, circlePos, radius):

        beginWithX, beginWithY = self.get_grid_point(circlePos[0] - radius - error, circlePos[1] - radius - error)
        endWithX, endWithY = self.get_grid_point(circlePos[0] + radius + error, circlePos[1] + radius + error)

        for x_idx in range(beginWithX, endWithX + 1):
            for y_idx in range(beginWithY, endWithY + 1):
                x, y = self.getCoordinatesByIdxs(x_idx, y_idx)
                dx = circlePos[0] - x
                dy = circlePos[1] - y
                if self.is_point_in_circle((x, y), circlePos, radius):
                    self.grid[x_idx, y_idx] = 2

    def paintFrameWithCircleReworked(self, circlePos, radius):

        beginWithX, beginWithY = self.get_grid_point(circlePos[0] - radius - error, circlePos[1] - radius - error)
        endWithX, endWithY = self.get_grid_point(circlePos[0] + radius + error, circlePos[1] + radius + error)

        for x_idx in range(beginWithX, endWithX):
            for y_idx in range(beginWithY, endWithY):
                if self.grid[x_idx, y_idx] == 2:
                    continue
                x, y = self.getCoordinatesByIdxs(x_idx, y_idx)
                dx = circlePos[0] - x
                dy = circlePos[1] - y
                if self.is_point_in_circle((x, y), circlePos, radius):
                    self.grid[x_idx, y_idx] = 2

    def paintFrameWithCircleReworked2(self, circlePos, radius):

        countOfColumns = self.get_grid_point(0, circlePos[1] + radius)[1] - self.get_grid_point(0, circlePos[1])[1]

        maxValue = radius
        startX = self.get_grid_point(circlePos[0] - maxValue, 0, only_x=True)[0]
        endX, y = self.get_grid_point(circlePos[0] + maxValue, circlePos[1])

        self.grid[startX: endX, y] = 2

        for i in range(1, countOfColumns):
            maxValue = math.cos(math.pi * i / countOfColumns) * radius
            startX = self.get_grid_point(circlePos[0] - maxValue, 0, only_x=True)[0]
            endX = self.get_grid_point(circlePos[0] + maxValue, 0, only_x=True)[0]

            self.grid[startX: endX, y + i] = 2
            self.grid[startX: endX, y - i] = 2

    def clearConture(self):
        for x_idx in range(len(self.grid)):
            for y_idx in range(len(self.grid[x_idx])):
                if self.grid[x_idx, y_idx] == 1:
                    self.grid[x_idx, y_idx] = 0

    def countWaterNeighbors(self, x_idx, y_idx):
        neighborsCount = 0

        maxX = len(self.grid)
        maxY = len(self.grid[0])

        if (x_idx - 1 >= 0 and y_idx - 1 >= 0):
            if self.grid[x_idx - 1, y_idx - 1] == 2: neighborsCount += 1
        if (y_idx - 1 >= 0):
            if self.grid[x_idx, y_idx - 1] == 2: neighborsCount += 1
        if (x_idx + 1 < maxX and y_idx - 1 >= 0):
            if self.grid[x_idx + 1, y_idx - 1] == 2: neighborsCount += 1

        if (x_idx - 1 >= 0):
            if self.grid[x_idx - 1, y_idx] == 2: neighborsCount += 1
        if (x_idx + 1 < maxX):
            if self.grid[x_idx + 1, y_idx] == 2: neighborsCount += 1

        if (x_idx - 1 >= 0 and y_idx + 1 < maxY):
            if self.grid[x_idx - 1, y_idx + 1] == 2: neighborsCount += 1
        if (y_idx + 1 < maxY):
            if self.grid[x_idx, y_idx + 1] == 2: neighborsCount += 1
        if (x_idx + 1 < maxX and y_idx + 1 < maxY):
            if self.grid[x_idx + 1, y_idx + 1] == 2: neighborsCount += 1

        return neighborsCount

    def countMarkedPoint(self, x_idx, y_idx, mark):
        neighborsCount = 0

        maxX = len(self.grid)
        maxY = len(self.grid[0])

        if (x_idx - 1 >= 0 and y_idx - 1 >= 0):
            if self.grid[x_idx - 1, y_idx - 1] == mark: neighborsCount += 1
        if (y_idx - 1 >= 0):
            if self.grid[x_idx, y_idx - 1] == mark: neighborsCount += 1
        if (x_idx + 1 < maxX and y_idx - 1 >= 0):
            if self.grid[x_idx + 1, y_idx - 1] == mark: neighborsCount += 1

        if (x_idx - 1 >= 0):
            if self.grid[x_idx - 1, y_idx] == mark: neighborsCount += 1
        if (x_idx + 1 < maxX):
            if self.grid[x_idx + 1, y_idx] == mark: neighborsCount += 1

        if (x_idx - 1 >= 0 and y_idx + 1 < maxY):
            if self.grid[x_idx - 1, y_idx + 1] == mark: neighborsCount += 1
        if (y_idx + 1 < maxY):
            if self.grid[x_idx, y_idx + 1] == mark: neighborsCount += 1
        if (x_idx + 1 < maxX and y_idx + 1 < maxY):
            if self.grid[x_idx + 1, y_idx + 1] == mark: neighborsCount += 1

        return neighborsCount

    def hasAtomsInFrame(self):
        return len(self.atomsInFrame) > 0

    def findFinalConture(self):
        for x_idx in range(len(self.grid)):
            for y_idx in range(len(self.grid[x_idx])):
                if self.grid[x_idx, y_idx] == 0:
                    neighbors = self.countWaterNeighbors(x_idx, y_idx)
                    if 1 < neighbors:
                        self.grid[x_idx, y_idx] = 3

    def findWaterOffset(self):
        for offsetConture in range(math.ceil(water_radius / self.grid_step)):
            for x_idx in range(len(self.grid)):
                for y_idx in range(len(self.grid[x_idx])):
                    if self.grid[x_idx, y_idx] == 0:
                        neighbors = self.countMarkedPoint(x_idx, y_idx, offsetConture + 3)
                        if 1 < neighbors:
                            self.grid[x_idx, y_idx] = offsetConture + 4

        for x_idx in range(len(self.grid)):
            for y_idx in range(len(self.grid[x_idx])):
                if self.grid[x_idx, y_idx] > 3:
                    self.grid[x_idx, y_idx] = 4

    def paintFrames(self):
        if len(self.atomsInFrame) == 0:
            self.grid[:, :] = 2
            return

        firstAtom = self.atomsInFrame[0]
        self.min_x_for_atom = firstAtom.x - firstAtom.radius
        self.max_x_for_atom = firstAtom.x + firstAtom.radius
        self.min_y_for_atom = firstAtom.y - firstAtom.radius
        self.max_y_for_atom = firstAtom.y + firstAtom.radius

        for atom in self.atomsInFrame:
            subminXForAtom = atom.x - atom.radius
            submaxXForAtom = atom.x + atom.radius
            subminYForAtom = atom.y - atom.radius
            submaxYForAtom = atom.y + atom.radius

            if subminXForAtom < self.min_x_for_atom:
                self.min_x_for_atom = subminXForAtom

            if subminYForAtom < self.min_y_for_atom:
                self.min_y_for_atom = subminYForAtom

            if submaxXForAtom > self.max_x_for_atom:
                self.max_x_for_atom = submaxXForAtom

            if submaxYForAtom > self.max_y_for_atom:
                self.max_y_for_atom = submaxYForAtom

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

        frame.paintFrames()
        cells_for_water = round(water_radius / frame.grid_step)

        print(idx, "out of:", count_of_levels)

        if frame.hasAtomsInFrame():
            for x_idx in range(frame.min_x_for_atom - cells_for_water - 1, frame.max_x_for_atom + cells_for_water + 1,
                               step_water_molecule):

                # print(x_idx, "out of:", len(current_frame.grid))
                for y_idx in range(frame.min_y_for_atom - cells_for_water - 1, frame.max_y_for_atom + cells_for_water + 1,
                                   step_water_molecule):
                    x, y = frame.getCoordinatesByIdxs(x_idx, y_idx)
                    if not frame.isMoleculeContainContuer((x, y), water_radius):
                        frame.paintFrameWithCircleReworked((x, y), water_radius)

        frame.clearConture()
        frame.findFinalConture()
        for i in range(math.ceil(water_radius / frame.grid_step)):
            frame.findWaterOffset()

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
