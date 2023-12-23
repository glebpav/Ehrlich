import math
from functools import cached_property

from ehrlich import MoleculeSurface, get_rotated_vector, find_inside_cone_points, Point


class Segment:
    def __init__(self, center_point, points: list[Point], molecule):
        self.center_point = center_point
        self.points = points
        self.molecule = molecule

    @cached_property
    def amines(self):
        used_idxs = {}
        segment_counter = {}
        for point in self.points:
            if point.atom_idx is None:
                print("surface points must be projected")
                # raise Exception("surface points must be projected")
                return None
            acid = self.molecule.atoms[point.atom_idx].residue
            if acid not in segment_counter:
                segment_counter[acid] = 1
                used_idxs[acid] = [point.atom_idx]
            elif point.atom_idx not in used_idxs[acid]:
                segment_counter[acid] += 1
                used_idxs[acid].append(point.atom_idx)
        return segment_counter

    def amin_similarity(self, segment):
        counter_result = {}
        score = 0

        for acid, count in self.amines.items():
            if acid in list(segment.amines.keys()):
                counter_result[acid] = min(count, segment.amines[acid])
                score += counter_result[acid]
        return score, counter_result

# def get_target_fi()


def make_sphere_segments(surface: MoleculeSurface, area: float, k: float) -> list[Segment]:

    """
    :param surface: MoleculeSurface object
    :param area: area of each segment
    :param k: overlay of cone angles
    :return: list of found segments
    """

    fi = math.degrees(math.acos(1 - area / (2 * math.pi * surface.sphere_radius ** 2)))

    print(fi)

    central_angle = fi
    vertical_angle_step = fi * k
    horizontal_angle_step = fi * k

    segments_list = []

    start_point_idx = 0
    for point_idx, point in enumerate(surface.points):
        if point_idx == 0:
            continue
        if point.origin_coords[2] < surface.points[start_point_idx].origin_coords[2]:
            start_point_idx = point_idx

    min_z = surface.points[start_point_idx].origin_coords[2]

    center_vector = surface.points[start_point_idx].origin_coords
    count_of_vertical_steps = math.floor(180 / vertical_angle_step)
    count_of_horizontal_steps = math.floor(360 / horizontal_angle_step)
    vertical_angle_step = 180 / count_of_vertical_steps

    for vertical_step in range(count_of_vertical_steps + 1):
        modified_count_of_horizontal_steps = int(
            max(1, count_of_horizontal_steps * (1 - (abs(center_vector[2]) / abs(min_z)) ** 7)))
        horizontal_angle_step = 360 / modified_count_of_horizontal_steps
        for horizontal_step in range(int(modified_count_of_horizontal_steps)):
            segments_list.append(Segment(
                center_vector.copy(),
                [surface.points[idx] for idx in find_inside_cone_points(surface.points, center_vector, central_angle)],
                surface.molecule
            ))
            center_vector = get_rotated_vector(center_vector, horizontal_angle_step, 'z')
        center_vector = get_rotated_vector(center_vector, vertical_angle_step, 'x')

    return segments_list
