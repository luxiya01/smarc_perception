from dataclasses import dataclass
from enum import Enum
import json
import os
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import shutil
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


def annotate_objects(data_dir: str, annotation_dir: str,
                     object_range_json: str):
    with open(object_range_json, 'r') as f:
        object_range = json.load(f)

    for filename, object_range_dict in object_range.items():
        print(f'Annotating file {filename}...')

        dvsfile = DVSFile(os.path.join(data_dir, filename))
        if "Side.PORT" in object_range_dict.keys():
            object_range_dict[Side.PORT] = object_range_dict["Side.PORT"]
        if "Side.STARBOARD" in object_range_dict.keys():
            object_range_dict[
                Side.STARBOARD] = object_range_dict["Side.STARBOARD"]

        for side, pings in dvsfile.sss_pings.items():
            nadir_annotation_file = os.path.join(
                annotation_dir, f'{filename}.{side}.annotation')
            object_annotation_file = os.path.join(
                annotation_dir, f'{filename}.{side}.objects.annotation')

            # copy annotation if no need to check for objects
            if side not in object_range_dict.keys():
                shutil.copy(nadir_annotation_file, object_annotation_file)
                continue

            # get indices to annotate for objects
            object_indices = []
            for object_range in object_range_dict[side]:
                object_indices.extend(
                    list(range(object_range[0], object_range[1])))

            # load nadir annotations
            nadirs = []
            with open(nadir_annotation_file, 'r') as f:
                for line in f:
                    _, start_idx, end_idx = [
                        int(x) for x in line.split(' ')[:3]
                    ]
                    nadirs.append(BoundingBox(start_idx, end_idx))

            # construct new annotations (nadir + objects) and write to file
            object_annotation = open(object_annotation_file, 'w')
            for idx, ping in enumerate(pings):
                ping_annotation = {}
                ping_annotation[ObjectID.NADIR] = nadirs[idx]

                if idx in object_indices:
                    rope = annotate_rope(ping, nadirs[idx])
                    ping_annotation[ObjectID.ROPE] = rope

                object_annotation.write(
                    _format_annotation_for_ping(ping_annotation))
            object_annotation.close()


def plot_annotations(data_dir: str, annotation_dir: str) -> None:
    dvsfiles = sorted([
        os.path.join(data_dir, x) for x in os.listdir(data_dir)
        if x.split('.')[-1] == 'dvs'
    ])

    for dvsfile in dvsfiles:
        print(dvsfile)
        plot_annotation_for_file(dvsfile, annotation_dir)


def plot_annotation_for_file(dvs_filepath: str, annotation_dir: str) -> None:
    dvs_filename = os.path.split(dvs_filepath)[-1]
    dvsfile = DVSFile(dvs_filepath)

    colors = {
        ObjectID.NADIR.value: 'y',
        ObjectID.ROPE.value: 'r',
        ObjectID.BUOY.value: 'k'
    }

    annotations = {
        Side.PORT: {
            ObjectID.NADIR.value: {},
            ObjectID.BUOY.value: {},
            ObjectID.ROPE.value: {}
        },
        Side.STARBOARD: {
            ObjectID.NADIR.value: {},
            ObjectID.BUOY.value: {},
            ObjectID.ROPE.value: {}
        }
    }

    for side, side_annotation in annotations.items():
        annotation_file = os.path.join(
            annotation_dir, f'{dvs_filename}.{side}.objects.annotation')
        with open(annotation_file, 'r') as f:
            for i, line in enumerate(f):
                values = [int(x) for x in line.split(' ')]
                if len(values) >= 3:
                    side_annotation[values[0]][i] = BoundingBox(
                        values[1], values[2])
                if len(values) >= 6:
                    side_annotation[values[3]][i] = BoundingBox(
                        values[4], values[5])
                if len(values) >= 9:
                    side_annotation[values[6]][i] = BoundingBox(
                        values[7], values[8])

        _, ax = dvsfile.plot_one_side(side)
        for k, v in side_annotation.items():
            y = list(v.keys())
            x1 = [x.start_idx for x in v.values()]
            x2 = [x.end_idx for x in v.values()]
            ax.scatter(x=x1, y=y, s=2, c=colors[k])
            ax.scatter(x=x2, y=y, s=2, c=colors[k])
    plt.show()


def annotate_nadir_for_dvsfile(filename: str) -> None:
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
        max_window_len: int) -> Tuple[BoundingBox, List[int]]:

    window.append(bbox.end_idx)
    if len(window) >= max_window_len:
        window.pop(0)
    window_average = int(sum(window) / len(window))

    if abs(bbox.end_idx - window_average) > continuity_constraint:
        bbox.end_idx = window[-2]
    return bbox, window


def _format_annotation_for_ping(
        ping_annotation: Dict[ObjectID, BoundingBox]) -> str:
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
    min_buoy_width = 10
    buoy_width = 15
    bkps = window_sliding_segmentation(ping=ping,
                                       start_idx=40,
                                       end_idx=nadir.end_idx,
                                       width=buoy_width,
                                       n_bkps=2)

    # Check whether the segmentation is likely to be a buoy
    if bkps[1] - bkps[0] > buoy_width * 2 or bkps[1] - bkps[0] < min_buoy_width:
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
