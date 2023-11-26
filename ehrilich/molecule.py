import os.path

import numpy as np

from ehrilich.utils.plane_utils import delete_unimportant


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


class Molecule:
    def __init__(self, atoms: list):
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

    def __get_item(self, idx):
        pass

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

    # TODO: write realization
    def sparse(self):
        saved_idx = delete_unimportant(self.coords)
        new_molecule = Molecule([atom for atom_idx, atom in enumerate(self.atoms) if atom_idx in saved_idx])
        new_molecule.original_mol = self
        return new_molecule


# TODO: implement 2-nd parser type
def read_pdb(path):
    if not os.path.exists(path):
        raise FileNotFoundError

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
