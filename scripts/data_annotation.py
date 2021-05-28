from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import ruptures as rpt
from dvs_file_reader import SSSPing, DVSFile, Side


class ObjectID(Enum):
    """ObjectID for object detection"""
    NADIR = 0
    ROPE = 1
    BUOY = 2


@dataclass
class BoundingBox:
    """1D bounding box for an object"""
    start_idx: int
    end_idx: int


def plot_nadir_annotations(data_dir: str, annotation_dir: str):
    dvsfiles = [x for x in os.listdir(data_dir) if x.split('.')[-1] == 'dvs']

    for filename in dvsfiles:
        dvsfile = DVSFile(os.path.join(data_dir, filename))
        nadir = {Side.PORT: {}, Side.STARBOARD: {}}
        for side, nadir_dict in nadir.items():
            annotation_file = os.path.join(annotation_dir,
                                           f'{filename}.{side}.annotation')
            with open(annotation_file, 'r') as f:
                for i, line in enumerate(f):
                    nadir_dict[i] = [int(x) for x in line.split(' ')][-1]
            _, ax = dvsfile.plot_one_side(side)
            ax.scatter(x=list(nadir_dict.values()),
                       y=list(nadir_dict.keys()),
                       s=2,
                       c='y')
    plt.show()


def annotate_nadir_for_dvsfile(filename: str):
    dvsfile = DVSFile(filename)
    continuity_constraint = 10

    for side, pings in dvsfile.sss_pings.items():
        annotation = open(f'{filename}.{side}.annotation', 'w')
        print(
            f'Annotating file {filename}, side = {side}, length = {len(pings)}'
        )
        prev_nadir_window = []

        for index, ping in enumerate(pings):
            ping_annotation = {}
            if index % 200 == 0:
                print(f'\tindex = {index}')

            nadir = annotate_nadir(ping)
            nadir, prev_nadir_window = _check_bbox_for_continuity_and_update_moving_window(
                nadir,
                prev_nadir_window,
                continuity_constraint,
                max_window_len=50)
            ping_annotation[ObjectID.NADIR] = nadir

            annotation.write(_format_annotation_for_ping(ping_annotation))
    annotation.close()


def _check_bbox_for_continuity_and_update_moving_window(
    bbox: BoundingBox, window: List[int], continuity_constraint: int,
    max_window_len: int):

    window.append(bbox.end_idx)
    if len(window) >= max_window_len:
        window.pop(0)
    window_average = int(sum(window) / len(window))

    if abs(bbox.end_idx - window_average) > continuity_constraint:
        bbox.end_idx = window[-2]
    return bbox, window


def _format_annotation_for_ping(ping_annotation: Dict[ObjectID, BoundingBox]):
    annotation_str = []
    fmt = lambda k, v: f'{k.value} {v.start_idx} {v.end_idx}'
    for k, v in ping_annotation.items():
        annotation_str.append(fmt(k, v))
    annotation_str = ' '.join(annotation_str)
    return f'{annotation_str}\n'


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
