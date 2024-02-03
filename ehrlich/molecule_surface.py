import os
import pickle
import time
import sys
from stl import mesh
from functools import cached_property

if sys.version_info>=(3, 9):
    import importlib.resources as pkg_resources
else:
    import importlib_resources as pkg_resources

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
        self.bonds = bonds # [(1, 2, 4), (2, 5, 7, 7) ... ] - list[Tuple([int])]
        # self.bonds = self.parse_points_bonds()

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
            dists = {idx: get_dist(mol_point.coords, point.shrunk_coords)
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

    def spiral_search(self, start_point_idx):

        visited_points = [start_point_idx]
        previous_point_idx = start_point_idx
        yield previous_point_idx

        visited_points.append(self.bonds[previous_point_idx][0])
        previous_point_idx = self.bonds[previous_point_idx][0]
        yield previous_point_idx

        while True:

            possible_points = self.bonds[previous_point_idx]
            visited_neighbour_counter = {}
            distance_counter = {}
            cos_counter = {}

            for point_idx in possible_points:

                if point_idx in visited_points: continue

                vect1 = self.points[visited_points[-1]].shrunk_coords - self.points[visited_points[-2]].shrunk_coords
                vect2 = self.points[point_idx].shrunk_coords - self.points[visited_points[-1]].shrunk_coords
                cos = get_cos(vect1, vect2)
                if cos < 0: continue
                if not any(map(lambda value: (value in visited_points) and (value != previous_point_idx), self.bonds[point_idx])): continue
                cos_counter[point_idx] = cos

                visited_neighbour_counter[point_idx] = any(map(lambda value: (value in visited_points) and (value != previous_point_idx), self.bonds[point_idx]))
                print(visited_neighbour_counter[point_idx])
                distance_counter[point_idx] = get_dist(self.points[point_idx].shrunk_coords, self.points[visited_points[0]].shrunk_coords)

            if len(visited_neighbour_counter.keys()) == 0:
                print("this branch")
                cos_counter = {}
                for point_idx in possible_points:
                    if point_idx in visited_points: continue
                    visited_neighbour_counter[point_idx] = any(map(lambda value: (value in visited_points) and (value != previous_point_idx),self.bonds[point_idx]))
                    print(visited_neighbour_counter[point_idx])
                    distance_counter[point_idx] = get_dist(self.points[point_idx].shrunk_coords,
                                                           self.points[visited_points[0]].shrunk_coords)

                if len(visited_neighbour_counter.keys()) == 0: return

            if len(list(cos_counter.keys())) > 0:
                print("case 1")
                max_point_idx = list(distance_counter.keys())[0]
                min_dist = distance_counter.get(list(distance_counter.keys())[0])
                min_cos = cos_counter.get(list(cos_counter.keys())[0])

                for key in list(distance_counter.keys()):
                    # if visited_neighbour_counter[key] and distance_counter[key] < min_dist:
                    if visited_neighbour_counter[key] and cos_counter[key] > min_cos:
                        min_dist = distance_counter[key]
                        min_cos = cos_counter[key]
                        max_point_idx = key

            else:
                print("case 2")
                max_point_idx = list(distance_counter.keys())[0]
                min_dist = distance_counter.get(list(distance_counter.keys())[0])

                for key in list(distance_counter.keys()):
                    if visited_neighbour_counter[key] and distance_counter[key] < min_dist:
                        min_dist = distance_counter[key]
                        max_point_idx = key

            next_point = max_point_idx
            print(visited_neighbour_counter)
            print('next: ', next_point)
            previous_point_idx = next_point
            visited_points.append(previous_point_idx)
            yield previous_point_idx



def load_molecule_surface(path) -> MoleculeSurface:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def make_surface(molecule, d=0.6, e=0.99):
    path_to_pqr = ""
    for str_item in molecule.path_to_pdb.split(".")[0:-1]:
        path_to_pqr += str(str_item)

    # converting pdb to pqr format
    os.system(f"pdb2pqr --ff=AMBER {molecule.path_to_pdb} {path_to_pqr}.pqr")

    tms_mesh_pkg = pkg_resources.files("ehrlich")
    tms_mesh_path = tms_mesh_pkg.joinpath("TMSmesh2.1")
    # creating shrunk surface
    os.system(f"{tms_mesh_path} {path_to_pqr}.pqr {d} {e}")

    file_name = f"{path_to_pqr}.pqr-{d}_{e}.off_modify.off"
    with open(file_name) as f:
        lines = f.readlines()

    coords = np.array([list(map(float, lines[i].split())) for i in range(2, int(lines[1].split()[0]) + 2)])
    connections = np.array([list(map(int, lines[i].split()[1:])) for i in range(int(lines[1].split()[0]) + 2, len(lines))])

    os.remove(file_name)
    os.remove(f"{path_to_pqr}.pqr")

    # converting connections from triangles to adj list
    adj = [set() for _ in range(len(coords))]
    for line in connections:
        idx1, idx2, idx3 = line
        adj[idx1].update([idx2, idx3])
        adj[idx2].update([idx1, idx3])
        adj[idx3].update([idx1, idx2])

    # preparing data for stl format
    cube = mesh.Mesh(np.zeros(connections.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(connections):
        for j in range(3):
            cube.vectors[i][j] = coords[f[j], :]

    points = []
    for point_idx in range(len(coords)):
        points.append(Point(
            origin_coords=[0, 0, 0],
            shrunk_coords=coords[point_idx],
            neighbors_points_idx=adj[point_idx],
        ))

    # creating object of MoleculeSurface
    molecule_surface = MoleculeSurface(points, [tuple(s) for s in adj])
    molecule_surface.molecule = molecule

    return molecule_surface, cube
