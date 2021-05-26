import numpy as np
import ruptures as rpt
from dvs_file_reader import SSSPing
from typing import List


def window_sliding_segmentation(ping: SSSPing,
                                start_idx: int = 20,
                                end_idx: int = None,
                                width: int = 20,
                                model: str = 'l2',
                                n_bkps: int = 3) -> List[int]:
    """Use window sliding method to segment the input SSSPing
    into (n_bkps + 1) segments. Return the suggested break points."""
    if not end_idx:
        end_idx = len(ping.ping)

    signal = np.array(ping.ping)
    algo = rpt.Window(width=width, model=model).fit(signal[start_idx:end_idx])
    bkps = algo.predict(n_bkps=n_bkps)
    bkps = [bkps[i] + start_idx for i in range(len(bkps))]
    return bkps
