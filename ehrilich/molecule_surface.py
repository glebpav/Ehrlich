import math
import os.path
import pickle
import time

import numpy as np

from ehrilich.optimizer import Optimizer
from ehrilich.utils.geometry_utils import Sphere
from ehrilich.utils.math_utils import *


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
    def __init__(self, points, bounds=None):
        self.__molecule = None
        self.points = points
        self.bounds = bounds

    @property
    def molecule(self):
        return self.__molecule

    @molecule.setter
    def molecule(self, molecule):
        self.__molecule = molecule

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    def load(self, path):
        with open(path, 'rb') as handle:
            mol_surface = pickle.load(handle)
            self.molecule = mol_surface.molecule
            self.points = self.points
            self.bounds = self.bounds

    def project(self):
        if self.__molecule is None:
            raise RuntimeError("molecule is None")

        for point_idx, point in enumerate(self.points):
            dists = {idx: get_dist(mol_point.shrinked_coords, point.shrinked_coords)
                     for idx, mol_point in enumerate(self.molecule)}
            dists = {k: v for k, v in sorted(dists.items(), key=lambda item: item[1])}
            best_atom_idx = list(dists.keys())[0]
            point.atom_idx = best_atom_idx
            self.molecule[best_atom_idx] = point_idx


# done - Создает внутри сферу
# done - оптимизатор
# done - сжимает сферу.
# done - Для каждой точки сферы создает объект Point,
# done - записывает в него начальные и сжатые кооринаты и список её соседей.
# done - После создает объект MoleculeSurface на списке точек и свяей между ними.
# done - В поле __molecule записывает молекулу на которой сжималась сфера.
# done - Возвращает обект MoleculeSurfac.

def make_surface(molecule, split_times=6):
    # sphere creation
    molecule_radius = molecule.get_radius()
    sphere = Sphere(molecule_radius + 10)
    for _ in range(split_times):
        print(sphere.split())

    points = [Point(origin_coords=list.copy(sphere_point)) for sphere_point in sphere.V]

    # optimizer creation
    optimizer = Optimizer(
        atoms=molecule.get_coords(),
        labels=molecule.get_atoms_names(),
        V=sphere.V,
        adj=sphere.adj,
        gpu=True
    )

    start_time = time.time()

    optimizer.optimize(
        Nsteps=10000,
        time_step=1e-1,
        lambd=0.5,
        close_atoms_ratio=1.35,
        comp_f_ratio=0.3,
        # comp_f_ratio=1.,
        patience=50,
        last_weight=0.9,
        accuracy_degree=3,
        doorstep_accuracy=1e-7
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

    molecule_surface = MoleculeSurface(points)
    molecule_surface.molecule = molecule
    return molecule
