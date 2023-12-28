import pickle
import time
from functools import cached_property

from ehrlich import Molecule
from ehrlich.optimizer import Optimizer
from ehrlich.sphere import Sphere
from ehrlich.utils.math_utils import *


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
            return

        # point_idx = mol_surface.points.index(self)
        point_idx = -1
        for idx, point in enumerate(mol_surface.points):
            if point.origin_coords[0] == self.origin_coords[0] and point.origin_coords[1] == self.origin_coords[1]:
                point_idx = idx
                # print(point_idx)
                break
        if point_idx == -1:
            print("Error no such point")
            return

        env_by_levels = {level_idx: [] for level_idx in range(3)}
        used_points = [point_idx]
        for level_idx in range(3):
            adj_points_idxs = []
            for selected_point_idx in used_points:
                adj_points_idxs += [adj_point_idx for adj_point_idx in mol_surface.points[selected_point_idx].neighbors_points_idx if
                                    adj_point_idx not in used_points]

            adj_points_idxs = list(set(adj_points_idxs))

            # sorting by cw or ccw order
            for idx1 in range(1, len(adj_points_idxs)):
                for idx2 in range(idx1 + 1, len(adj_points_idxs)):
                    dist1 = get_dist(mol_surface.points[adj_points_idxs[idx1 - 1]].origin_coords,
                                     mol_surface.points[adj_points_idxs[idx2]].origin_coords)
                    dist2 = get_dist(mol_surface.points[adj_points_idxs[idx1 - 1]].origin_coords,
                                     mol_surface.points[adj_points_idxs[idx1]].origin_coords)
                    if dist1 < dist2:
                        temp = adj_points_idxs[idx1]
                        adj_points_idxs[idx1] = adj_points_idxs[idx2]
                        adj_points_idxs[idx2] = temp

            # finding ccw order
            vect1 = mol_surface.points[adj_points_idxs[0]].origin_coords - mol_surface.points[point_idx].origin_coords
            vect2 = mol_surface.points[adj_points_idxs[1]].origin_coords - mol_surface.points[point_idx].origin_coords
            res_vect = np.cross(vect1, vect2)
            center_vector = mol_surface.points[point_idx].origin_coords

            cos_a = np.dot(res_vect, center_vector) / (np.linalg.norm(res_vect) * np.linalg.norm(center_vector))

            # 1 2 3 4 ...
            if cos_a > 0:
                vectors_sequence = list(range(len(adj_points_idxs)))
            # n n-1 n-2 ...
            else:
                vectors_sequence = list(range(len(adj_points_idxs) - 1, -1, -1))

            for idx, vector_idx in enumerate(vectors_sequence):
                env_by_levels[level_idx].append(adj_points_idxs[vectors_sequence[idx]])
            used_points += adj_points_idxs

        norm_components = []
        for idx in range(1, len(env_by_levels[1])):
            vect1 = mol_surface.points[env_by_levels[1][idx - 1]].origin_coords - self.origin_coords
            vect2 = mol_surface.points[env_by_levels[1][idx]].origin_coords - self.origin_coords
            res_vect = np.cross(vect1, vect2)
            norm_components.append(get_norm(res_vect))
        avg_adj_vect = np.average(np.array(norm_components), axis=0)
        is_norm_inverted = get_cos(avg_adj_vect, self.origin_coords) > 0

        norm_components = []
        for i in range(1, 3):
            for idx in range(1, len(env_by_levels[i])):
                vect1 = mol_surface.points[env_by_levels[i][idx - 1]].shrunk_coords - self.shrunk_coords
                vect2 = mol_surface.points[env_by_levels[i][idx]].shrunk_coords - self.shrunk_coords
                res_vect = np.cross(vect1, vect2)
                norm_components.append(get_norm(res_vect))

        avg_adj_vect = np.average(np.array(norm_components), axis=0)
        norm_avg = np.array(get_norm(avg_adj_vect))

        if not is_norm_inverted:
            norm_avg *= -1

        return norm_avg


class MoleculeSurface:
    def __init__(self, points, bonds):
        self.__molecule = None
        self.points = points  # list[Points]
        # self.bonds = self.parse_points_bonds()
        self.bonds = bonds

    def parse_points_bonds(self):
        bonds = []
        for point in self.points:
            bonds.append(point.neighbors_points_idx)
        return bonds

    @property
    def molecule(self) -> Molecule:
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
            self.log(point_idx)
            dists = {idx: get_dist(mol_point.coordinates, point.shrunk_coords)
                     for idx, mol_point in enumerate(self.molecule)}
            dists = {k: v for k, v in sorted(dists.items(), key=lambda item: item[1])}
            best_atom_idx = list(dists.keys())[0]
            point.atom_idx = best_atom_idx
            self.molecule[best_atom_idx].point = point_idx

    def log(self, n):
        r = int(30 * n / len(self.points))
        print(f"\rProjecting: {n}/{len(self.points)} |{'=' * r}>{'.' * (30 - r)}|", end='')

    @cached_property
    def average_shrunk_edge_len(self):
        sum_len = 0
        count = 0
        for idx1, list_idxs in enumerate(self.bonds):
            for idx2 in list_idxs:
                sum_len += get_dist(self.points[idx1].shrunk_coords, self.points[idx2].shrunk_coords)
                count += 1
        return sum_len / count

    @cached_property
    def average_sphere_edge_len(self):
        sum_len = 0
        count = 0
        for idx1, list_idxs in enumerate(self.bonds):
            for idx2 in list_idxs:
                sum_len += get_dist(self.points[idx1].origin_coords, self.points[idx2].origin_coords)
                count += 1
        return sum_len / count

    def sphere_area(self, target_area):
        return target_area * (self.average_sphere_edge_len / self.average_sphere_edge_len) ** 2

    @cached_property
    def sphere_radius(self):
        norms = []
        for point in self.points:
            norms.append(np.linalg.norm(point.shrunk_coords))
        return sum(norms) / len(norms)


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
        atoms=molecule.get_coordinates(),
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
