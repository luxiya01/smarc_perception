from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import ruptures as rpt
from dvs_file_reader import SSSPing, Side, DVSFile, ObjectID, BoundingBox


def _update_moving_average(current_average: int, new_value: int, seq_len: int):
    """Calculate moving average online."""
    if current_average is None:
        return new_value
    return int(current_average + (new_value - current_average) / (seq_len))


def annotate_rope(ping: SSSPing, nadir: BoundingBox) -> BoundingBox:
    """Given the tentative nadir_annotation, provide tentative rope
    annotation by segmenting the nadir region."""
    bkps = window_sliding_segmentation(ping=ping,
                                       start_idx=40,
                                       end_idx=nadir.end_idx,
                                       width=4,
                                       n_bkps=1)
    return BoundingBox(start_idx=bkps[0] - 1, end_idx=bkps[0] + 1)


def annotate_buoy(ping: SSSPing, nadir: BoundingBox) -> BoundingBox:
    """Given the tentative nadir_annotation, provide tentative buoy
    annotation by segmenting the nadir region. Return None if no
    buoy detected."""
    buoy_width = 15
    bkps = window_sliding_segmentation(ping=ping,
                                       start_idx=40,
                                       end_idx=nadir.end_idx,
                                       width=buoy_width,
                                       n_bkps=2)

    # Check whether the segmentation is likely to be a buoy
    if bkps[1] - bkps[0] > buoy_width * 2:
        return None
    return BoundingBox(start_idx=bkps[0], end_idx=bkps[1])


def annotate_nadir(ping: SSSPing) -> BoundingBox:
    """Use window sliding segmentation to provide tentative
    nadir location annotation. Returns the start and end index
    of nadir region, assuming nadir always starts at index 0."""
    bkps = window_sliding_segmentation(ping=ping,
                                       start_idx=100,
                                       width=100,
                                       n_bkps=1)
    return BoundingBox(start_idx=0, end_idx=bkps[0])


def window_sliding_segmentation(ping: SSSPing,
                                start_idx: int = 20,
                                end_idx: int = None,
                                width: int = 20,
                                model: str = 'l2',
                                n_bkps: int = 3) -> List[int]:
    """Use window sliding method to segment the input SSSPing
    into (n_bkps + 1) segments. Return the suggested break points."""
    signal = ping.get_ping_array(normalised=True)
    if not end_idx:
        end_idx = signal.shape[0]

    algo = rpt.Window(width=width, model=model).fit(signal[start_idx:end_idx])
    bkps = algo.predict(n_bkps=n_bkps)
    bkps = [bkps[i] + start_idx for i in range(len(bkps))]
    return bkps
