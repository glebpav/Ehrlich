import time
from typing import Tuple, List, Callable

import numpy as np

from ehrlich.segment import Segment

DEFAULT_AMIN_THRESHOLD = 0.1
DEFAULT_GEOM_THRESHOLD = 0.5


def filter_paris_data(
        pairs: List[Tuple[Segment, Segment]],
        extended_filter_function: Callable[[Segment, Segment], Tuple[bool, float]],
        top_paris: int,
        reverse_order_sort: bool = False,
) -> List[Tuple[Segment, Segment]]:

    pair_value_list = np.zeros(len(pairs))
    is_pair_passed_list = np.array(([False] * len(pairs)))

    for pair_idx, (segment1, segment2) in enumerate(pairs):
        is_pair_passed_list[pair_idx], pair_value_list[pair_idx] = extended_filter_function(segment1, segment2)

    after_filter_size = min(top_paris, is_pair_passed_list.sum())

    if after_filter_size == 0:
        raise Exception("Segments are too different")

    geometry_top_paris_idxs = np.argpartition(
        pair_value_list,
        -after_filter_size if reverse_order_sort else after_filter_size
    )[:after_filter_size]

    top_pairs = [pairs[idx] for idx in geometry_top_paris_idxs]

    return top_pairs


def default_geom_descr_filter(segment1: Segment, segment2: Segment) -> (bool, float):
    value = abs(segment1.concavity - segment2.concavity)
    return value < DEFAULT_GEOM_THRESHOLD, value


def default_amin_descr_filter(segment1: Segment, segment2: Segment) -> (bool, float):
    value = segment1.amin_similarity(segment2)
    return value > DEFAULT_AMIN_THRESHOLD, value


def filter_segments_by_descriptors(
        pairs: List[Tuple[Segment, Segment]],
        descr_geom_filter: Callable[[Segment, Segment], Tuple[bool, float]] = default_geom_descr_filter,
        descr_gtop: int = 15,
        descr_amin_filter: Callable[[Segment, Segment], Tuple[bool, float]] = default_amin_descr_filter,
        descr_atop: int = 10
) -> np.array:

    descriptors_geometry_top = filter_paris_data(pairs, descr_geom_filter, descr_gtop)
    descriptors_amin_top = filter_paris_data(descriptors_geometry_top, descr_amin_filter, descr_atop, True)

    return descriptors_amin_top


def default_segment_align_filter(segment1: Segment, segment2: Segment) -> (bool, float):
    return 1, True


def default_molecule_align_filter(segment1: Segment, segment2: Segment) -> (bool, float):
    return 1, True


def filter_segments_by_alignment(
        pairs: List[Tuple[Segment, Segment]],
        align_segment_filter: Callable[[Segment, Segment], Tuple[bool, float]] = default_segment_align_filter,
        align_segment_top: int = 5,
        align_mol_filter: Callable[[Segment, Segment], Tuple[bool, float]] = default_molecule_align_filter,
        align_mtop: int = 3,
) -> np.array:

    align_segment_top = filter_paris_data(pairs, align_segment_filter, align_segment_top)
    align_molecule_top = filter_paris_data(align_segment_top, align_mol_filter, align_mtop)

    return align_molecule_top















