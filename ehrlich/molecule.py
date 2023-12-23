import math
import os.path
from decimal import Decimal

import numpy as np

from ehrlich.utils.math_utils import double_range
from ehrlich.utils.plane_utils import GirdPoint, Frame


class Atom:
    """
    Class that stores atom information
    """

    def __init__(self, idx, name, residue, residue_num, x, y, z):
        """
        :param idx: atom idx int pdb
        :param name: name as in .pdb
        :param residue: name of the residue (amino acid)
        :param residue_num: number of residue
        :param x: x coordinate of atom
        :param y: y coordinate of atom
        :param z: z coordinate of atom
        """
        self.name = name
        self.coords = np.array([x, y, z])
        self.idx = idx
        self.residue = residue
        self.residue_num = residue_num
        self.point = None


class Sparsify:
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

    def delete_unimportant(self, coords):
        coords = np.array([[atom[0], atom[1], atom[2]] for atom in coords])
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        max_radius = self.MAX_VWR

        max_z = max(z) + max_radius + self.water_radius * 2
        min_z = min(z) - max_radius - self.water_radius * 2

        min_x = min(x) - max_radius - self.water_radius * 2
        max_x = max(x) + max_radius + self.water_radius * 2

        min_y = min(y) - max_radius - self.water_radius * 2
        max_y = max(y) + max_radius + self.water_radius * 2

        print(max_z, min_z, max_radius)

        points = [GirdPoint(float(coord[0]), float(coord[1]), float(coord[2]), self.atom_radius) for coord in coords]

        for idx, point in enumerate(points):
            point.idx = idx

        levels = []
        for currentZ in double_range(Decimal(min_z), Decimal(max_z), self.z_step):

            current_frame = Frame(max_x - min_x + self.error, max_y - min_y + self.error, self.lower_step_x, min_x,
                                  min_y, self.count_of_points_for_circle, self.error, self.water_radius)

            print(float('{:.2f}'.format(currentZ)), 'out of', float('{:.2f}'.format(max_z)))

            for point in points:
                if (point.z > currentZ > (point.z - point.radius)) or (point.z < currentZ < point.z + point.radius):
                    new_radius = ((point.radius ** 2) - ((currentZ - point.z) ** 2)) ** 0.5
                    current_frame.process_circle((point.x, point.y), new_radius)

            levels.append(current_frame)

        count_of_levels = len(levels)
        for idx, frame in enumerate(levels):

            frame.paint_frames()
            cells_for_water = round(self.water_radius / frame.grid_step)

            print(idx, "out of:", count_of_levels)

            if frame.has_atoms_in_frame():
                for x_idx in range(frame.min_x_for_atom - cells_for_water - 1,
                                   frame.max_x_for_atom + cells_for_water + 1,
                                   self.step_water_molecule):

                    for y_idx in range(frame.min_y_for_atom - cells_for_water - 1,
                                       frame.max_y_for_atom + cells_for_water + 1,
                                       self.step_water_molecule):
                        x, y = frame.get_coordinates_by_idxs(x_idx, y_idx)
                        if not frame.is_molecule_contain_contur((x, y), self.water_radius):
                            frame.paint_frame_with_circle_reworked((x, y), self.water_radius)

            frame.clear_contur()
            frame.find_final_contur()
            for i in range(math.ceil(self.water_radius / frame.grid_step)):
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

    def sparse(self):
        saved_idx = self.delete_unimportant(self.coords)
        new_molecule = Molecule([atom for atom_idx, atom in enumerate(self.atoms) if atom_idx in saved_idx])
        new_molecule.original_mol = self
        return new_molecule


class Molecule(Sparsify):
    def __init__(self, atoms):
        self.atoms = atoms.copy()
        self.coords = self.__get_coords(atoms)
        self.original_mol = None

    def __iter__(self):
        return iter(self.atoms)

    def __getitem__(self, item: int):
        return self.atoms[item]

    @staticmethod
    def __get_coords(atoms):
        coords_list = [atom.coords for atom in atoms]
        return np.array(coords_list)

    def get_radius(self):
        max_atom_distance = 0.
        for atom in self.atoms:
            max_atom_distance = max(np.linalg.norm(atom.coords), max_atom_distance)
        return max_atom_distance

    def get_coords(self):
        coords = [atom.coords for atom in self.atoms]
        return np.array(coords)

    def get_atoms_names(self):
        names = [atom.name for atom in self.atoms]
        return names

    """# TODO: write realization
    def sparse(self):
        saved_idx = delete_unimportant(self.coords)
        new_molecule = Molecule([atom for atom_idx, atom in enumerate(self.atoms) if atom_idx in saved_idx])
        new_molecule.original_mol = self
        return new_molecule"""


# TODO: implement 2-nd parser type
def read_pdb(path):

    with open(path) as f:
        lines = f.readlines()

    atoms_list = []

    for line in lines:
        if not line.startswith('ATOM'):
            continue

        split_line = line.split()

        atom_idx = int(split_line[1])
        full_atom_name = split_line[2]
        atom_name = full_atom_name[0]
        amino_acid = split_line[3]
        amino_acid_idx = split_line[5]
        x = float(split_line[6])
        y = float(split_line[7])
        z = float(split_line[8])

        atoms_list.append(Atom(atom_idx, atom_name, amino_acid, amino_acid_idx, x, y, z))

    return Molecule(atoms_list)
