import math
from functools import cached_property

from ehrilch import MoleculeSurface, get_rotated_vector, find_inside_cone_points


class Segment:
    def __init__(self, center_point, points):
        self.center_point = center_point
        self.points = points
        self.molecule = None  # implement logic
        self.segment_counter = None

    @cached_property
    def amines(self):
        used_idxs = {}
        segment_counter = {}
        for point in self.points:
            if point.acid not in segment_counter:
                segment_counter[point.acid] = 1
                used_idxs[point.acid] = [point.atom_point_idx]
            else:
                if point.atom_point_idx not in used_idxs[point.acid]:
                    segment_counter[point.acid] += 1
                    used_idxs[point.acid].append(point.atom_point_idx)
        return segment_counter

    def amin_similarity(self, segment):
        counter_result = {}
        score = 0

        for acid, count in self.amines().items():
            if acid in list(segment.amines().keys()):
                counter_result[acid] = min(count, segment.amines()[acid])
                score += counter_result[acid]
        return score, counter_result


def make_sphere_segments(surface: MoleculeSurface, fi: float, k: float) -> list[Segment]:

    """
    :param surface: MoleculeSurface object
    :param fi: cone angle
    :param k: overlay of cone angles
    :return: list of found segments
    """

    central_angle = fi
    vertical_angle_step = fi * k
    horizontal_angle_step = fi * k

    segments_list = []

    start_point_idx = 0
    for point_idx, point in enumerate(surface.points):
        if point_idx == 0:
            continue
        if point.origin_coords[2] < surface.points[start_point_idx].origin_coords[2]:
            start_point_idx = point.idx

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
                find_inside_cone_points(surface.points, center_vector, central_angle)
            ))
            center_vector = get_rotated_vector(center_vector, horizontal_angle_step, 'z')
        center_vector = get_rotated_vector(center_vector, vertical_angle_step, 'x')

    return segments_list
