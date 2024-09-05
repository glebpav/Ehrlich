import itertools
import time
from typing import Tuple, List, Callable

import numpy as np

from ehrlich import MoleculeStructure
from ehrlich.alignment import Alignment
from ehrlich.segment import Segment

DEFAULT_AMIN_THRESHOLD = 0.1
DEFAULT_GEOM_THRESHOLD = 0.5


def filter_descriptor_paris_data(
        pairs: List[Tuple[Segment, Segment]],
        extended_filter_function: Callable[[Segment, Segment], Tuple[bool, float]],
        top_paris: int,
        reverse_order_sort: bool = False,
) -> List[Tuple[Segment, Segment]]:

    if len(pairs) == 0:
        return list()

    pair_value_list = np.zeros(len(pairs))
    is_pair_passed_list = np.array(([False] * len(pairs)))

    for pair_idx, (segment1, segment2) in enumerate(pairs):
        is_pair_passed_list[pair_idx], pair_value_list[pair_idx] = extended_filter_function(segment1, segment2)

    after_filter_size = min(top_paris, is_pair_passed_list.sum())

    if after_filter_size == 0:
        return list()

    geometry_top_paris_idxs = np.argpartition(
        pair_value_list,
        -after_filter_size if reverse_order_sort else after_filter_size
    )[:after_filter_size]

    """geometry_top_paris_idxs = np.argsort(pair_value_list)
    if reverse_order_sort:
        geometry_top_paris_idxs = np.flip(geometry_top_paris_idxs)"""

    top_pairs = [pairs[idx] for idx in geometry_top_paris_idxs[:after_filter_size]]

    return top_pairs


def filter_alignment_pairs_data(
        alignment_list: List[Alignment],
        extended_filter_function: Callable[[Alignment], Tuple[bool, float]],
        top_paris: int,
) -> List[Alignment]:

    if len(alignment_list) == 0:
        return list()

    pair_value_list = np.zeros(len(alignment_list))
    is_pair_passed_list = np.array(([False] * len(alignment_list)))

    for alignment_idx, alignment in enumerate(alignment_list):
        is_pair_passed_list[alignment_idx], pair_value_list[alignment_idx] = extended_filter_function(alignment)

    after_filter_size = min(top_paris, is_pair_passed_list.sum())

    if after_filter_size == 0:
        return list()

    geometry_top_paris_idxs = np.argsort(pair_value_list)
    top_pairs = [alignment_list[idx] for idx in geometry_top_paris_idxs[:after_filter_size]]

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
) -> List[Tuple[Segment, Segment]]:

    descriptors_geometry_top = filter_descriptor_paris_data(pairs, descr_geom_filter, descr_gtop)
    descriptors_amin_top = filter_descriptor_paris_data(descriptors_geometry_top, descr_amin_filter, descr_atop, True)

    return descriptors_amin_top


def default_segment_align_filter(alignment: Alignment) -> (bool, float):
    return True, alignment.geom_dist


def default_molecule_align_filter(alignment: Alignment) -> (bool, float):
    return True, alignment.geom_dist


def filter_segments_by_alignment(
        pairs: List[Tuple[Segment, Segment]],
        align_segment_filter: Callable[[Alignment], Tuple[bool, float]] = default_segment_align_filter,
        align_segment_top: int = 5,
        align_mol_filter: Callable[[Alignment], Tuple[bool, float]] = default_molecule_align_filter,
        align_mtop: int = 3,
) -> List[Alignment]:

    alignment_list = [
        segment1.align(segment2, icp_iterations=30)
        for (segment1, segment2) in pairs
    ]

    alignment_list = filter_alignment_pairs_data(alignment_list, align_segment_filter, align_segment_top)

    alignment_list = [
        alignment.segment1.mol_align(alignment.segment2, icp_iterations=10)
        for alignment in alignment_list
    ]

    alignment_list = filter_alignment_pairs_data(alignment_list, align_mol_filter, align_mtop)

    return alignment_list


def combine_segments(segments1: List[Segment], segments2: List[Segment]) -> List[Tuple[Segment, Segment]]:
    return list(itertools.product(segments1, segments2))


def find_fit_segments(segments1, segments2, **kwargs) -> List[Alignment]:
    segments_pairs = combine_segments(segments1, segments2)
    after_desc_pairs = filter_segments_by_descriptors(segments_pairs)
    after_align_pairs = filter_segments_by_alignment(after_desc_pairs, align_mtop=kwargs['topk'])
    return after_align_pairs


def find_best_mol_alignment(
        struct1: MoleculeStructure,
        struct2: MoleculeStructure,
        topk: int = 3
) -> List[Alignment]:
    return find_fit_segments(struct1.segments, struct2.segments, topk=topk)













