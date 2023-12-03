import pickle
import time
from functools import cached_property

from ehrilch.optimizer import Optimizer
from ehrilch.sphere import Sphere
from ehrilch.utils.math_utils import *


class Point:
    def __init__(self, origin_coords=None, shrunk_coords=None, atom_idx=None, neighbors_points_idx=None):
        self.origin_coords = origin_coords
        self.shrunk_coords = shrunk_coords
        self.atom_idx = atom_idx
        self.neighbors_points_idx = neighbors_points_idx
        self.norm = None

    # TODO: implement this method
    def compute_norm(self, mol_surface):
        if self.neighbors_points_idx is None:
            raise RuntimeError("point has no neighbours")

        norm_components = []
        # first env
        for neighbor_point_idx in self.neighbors_points_idx:
            vect1 = mol_surface.points[neighbor_point_idx].shrunk_coords - self.shrunk_coords
            norm_components.append(get_norm(vect1))

        # second env
        for neighbor_point_idx in self.neighbors_points_idx:
            for second_neighbor_point_idx in mol_surface.points[neighbor_point_idx]:
                if second_neighbor_point_idx in self.neighbors_points_idx:
                    continue
                vect1 = mol_surface.points[second_neighbor_point_idx].shrunk_coords - self.shrunk_coords
                norm_components.append(vect1)

        avg_adj_vect = np.average(np.array(norm_components), axis=0)
        self.norm = get_norm(avg_adj_vect)
        return self.norm


class MoleculeSurface:
    def __init__(self, points, bonds):
        self.__molecule = None
        self.points = points
        # self.bonds = self.parse_points_bonds()
        self.bonds = bonds

    def parse_points_bonds(self):
        bonds = []
        for point in self.points:
            bonds.append(point.neighbors_points_idx)
        return bonds

    @property
    def molecule(self):
        return self.__molecule

    @molecule.setter
    def molecule(self, molecule):
        self.__molecule = molecule

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def project(self):
        if self.__molecule is None:
            raise RuntimeError("molecule is None")
        print("here")
        for point_idx, point in enumerate(self.points):
            print(f"passed: {point_idx} of {len(self.points)}")
            dists = {idx: get_dist(mol_point.coords, point.shrunk_coords)
                     for idx, mol_point in enumerate(self.molecule)}
            dists = {k: v for k, v in sorted(dists.items(), key=lambda item: item[1])}
            best_atom_idx = list(dists.keys())[0]
            point.atom_idx = best_atom_idx
            self.molecule[best_atom_idx].point = point_idx

    @cached_property
    def get_average_shrunk_edge_len(self):
        sum_len = 0
        count = 0
        for idx1, list_idxs in enumerate(self.bonds):
            for idx2 in list_idxs:
                sum_len += get_dist(self.points[idx1].shrunk_coords, self.points[idx2].shrunk_coords)
                count += 1
        return sum_len / count

    @cached_property
    def get_average_sphere_edge_len(self):
        sum_len = 0
        count = 0
        for idx1, list_idxs in enumerate(self.bonds):
            for idx2 in list_idxs:
                sum_len += get_dist(self.points[idx1].origin_coords, self.points[idx2].origin_coords)
                count += 1
        return sum_len / count


def load_molecule_surface(path) -> MoleculeSurface:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def make_surface(
        molecule,
        split_times=6,
        n_steps=10000,
        time_step=1e-1,
        lambda_k=0.5,
        close_atoms_ratio=1.35,
        comp_f_ratio=0.3,
        # comp_f_ratio=1.,
        patience=50,
        last_weight=0.9,
        accuracy_degree=3,
        doorstep_accuracy=1e-7,
        gpu=False
):
    # sphere creation
    molecule_radius = molecule.get_radius()
    sphere = Sphere(molecule_radius + 10)
    for _ in range(split_times):
        print(sphere.split())

    # optimizer creation
    optimizer = Optimizer(
        atoms=molecule.get_coords(),
        labels=molecule.get_atoms_names(),
        V=sphere.V,
        adj=sphere.adj,
        gpu=gpu
    )

    start_time = time.time()

    optimizer.optimize(
        Nsteps=n_steps,
        time_step=time_step,
        lambda_k=lambda_k,
        close_atoms_ratio=close_atoms_ratio,
        comp_f_ratio=comp_f_ratio,
        # comp_f_ratio=1.,
        patience=patience,
        last_weight=last_weight,
        accuracy_degree=accuracy_degree,
        doorstep_accuracy=doorstep_accuracy
    )
    # optional
    print(f"\nTime passed: {time.time() - start_time}")

    shrunk_coords = optimizer.V.cpu().numpy()

    points = []
    for point_idx in range(len(shrunk_coords)):
        points.append(Point(
            origin_coords=sphere.V[point_idx],
            shrunk_coords=shrunk_coords[point_idx],
            neighbors_points_idx=sphere.adj[point_idx],
        ))

    molecule_surface = MoleculeSurface(points, sphere.adj)
    molecule_surface.molecule = molecule
    return molecule_surface
