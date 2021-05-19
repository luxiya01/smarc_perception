from enum import Enum
from dataclasses import dataclass
import utils
from typing import List, BinaryIO, Tuple, Dict
import struct
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


class Side(Enum):
    """The side-scan sonar ping side (port or starboard)"""
    PORT = 0
    STARBOARD = 1


@dataclass
class SSSPing:
    """A side-scan sonar ping."""
    lat: float  #Ctype double in V1_Position in deepvision_sss_driver
    lon: float  #Ctype double in V1_Position deepvision_sss_driver
    speed: float
    heading: float
    side: Side
    ping: List[int]

    def plot(self) -> None:
        plt.figure()
        plt.plot(self.ping,
                 linestyle='-',
                 marker='o',
                 markersize=1,
                 linewidth=.5)
        plt.title(f'SSS Ping, side = {self.side}')


@dataclass
class DVSFileHeader:
    """DVS File header. See details in deepvision_sss_driver."""
    version: int = -1
    sample_res: float = -1
    line_rate: float = -1
    n_samples: int = -1
    left: bool = False
    right: bool = False


class DVSFile:
    def __init__(self, filename: str):
        self.filename = filename

        # Placeholder attributes, filled by _parse_file()
        self.header = None
        self.sss_pings = {Side.PORT: [], Side.STARBOARD: []}

        self._parse_file()

    def _parse_file(self):
        with open(self.filename, 'rb') as f:
            self.header = self._parse_header(f)
            while True:
                try:
                    ping = self._parse_ping(f)
                    for key, value in ping.items():
                        self.sss_pings[key].append(value)
                except struct.error as e:
                    pointer_pos = f.tell()
                    file_size = Path(self.filename).stat().st_size
                    print(f'Current pointer position: {pointer_pos}')
                    print(f'Total file size: {file_size}')
                    print(f'Remaining bytes: {file_size - pointer_pos}')
                    print(f'Parsing completed: {e}')
                    return

    def _parse_ping(self, fileobj: BinaryIO) -> Dict[Side, SSSPing]:
        """Read one side-scan ping from the fileobj. Note that one ping
        may consists of two channels (port and starboard)."""
        lat = utils.unpack_struct(fileobj, struct_type='double')
        lon = utils.unpack_struct(fileobj, struct_type='double')
        speed = utils.unpack_struct(fileobj, struct_type='float')
        heading = utils.unpack_struct(fileobj, struct_type='float')

        ping = dict()
        if self.header.left:
            left_channel = utils.unpack_channel(
                fileobj, channel_size=self.header.n_samples)
            ping[Side.PORT] = SSSPing(lat=lat,
                                      lon=lon,
                                      speed=speed,
                                      heading=heading,
                                      side=Side.PORT,
                                      ping=left_channel)
        if self.header.right:
            right_channel = utils.unpack_channel(
                fileobj, channel_size=self.header.n_samples)
            ping[Side.STARBOARD] = SSSPing(lat=lat,
                                           lon=lon,
                                           speed=speed,
                                           heading=heading,
                                           side=Side.STARBOARD,
                                           ping=right_channel)
        return ping

    def _parse_header(self, fileobj: BinaryIO) -> DVSFileHeader:
        """Read version and V1_FileHeader from the file object"""
        header = DVSFileHeader()
        header.version = utils.unpack_struct(fileobj, struct_type='uint')
        header.sample_res = utils.unpack_struct(fileobj, struct_type='float')
        header.line_rate = utils.unpack_struct(fileobj, struct_type='float')
        header.n_samples = utils.unpack_struct(fileobj, struct_type='int')
        header.left = utils.unpack_struct(fileobj, struct_type='bool')
        header.right = utils.unpack_struct(fileobj, struct_type='bool')

        # DVSFileHeader object is 16 bytes, although all fields together adds up to
        # 14 bytes. The 2 extra bytes are for probably for data structure alignment
        fileobj.read(2)

        return header

    def _get_pings_from_one_side(self, side: Side, start_idx: int,
                                 end_idx) -> np.ndarray:
        pings = []
        for i in range(start_idx, end_idx):
            pings.append(self.sss_pings[side][i].ping)
        return np.array(pings)

    def _imshow(self, sss_pings: np.ndarray, start_idx: int, end_idx: int,
                title: str, figsize: tuple) -> None:
        """Plot multiple SSSPings as an heatmap."""
        num_pings, num_channels = sss_pings.shape

        plt.figure(figsize=figsize)
        plt.imshow(sss_pings,
                   origin='lower',
                   extent=(0, num_channels, start_idx, end_idx))
        plt.title(title)

    def plot_one_side(
            self,
            side: Side,
            start_idx: int = 0,
            end_idx: int = None,
            figsize: tuple = (5, 10)) -> np.ndarray:
        """Plot sss pings between (start_idx, end_idx) from the requested side
        if exists."""
        if side not in self.sss_pings.keys():
            raise ValueError(
                f'Side {side} does not exist. Available sides: {self.sss_pings.keys()}'
            )
        if not end_idx:
            end_idx = len(self.sss_pings[side])
        side_pings = self._get_pings_from_one_side(side, start_idx, end_idx)

        title = f'SSS pings from {side} of {self.filename}'
        self._imshow(side_pings, start_idx, end_idx, title, figsize)

        return side_pings

    def plot(self,
             start_idx: int = 0,
             end_idx: int = None,
             figsize: tuple = (10, 20)) -> np.ndarray:
        """Plot all sss pings in the DVSFile"""
        if self.header.right and not self.header.left:
            return self.plot_one_side(side=Side.STARBOARD,
                                      start_idx=start_idx,
                                      end_idx=end_idx)
        if self.header.left and not self.header.right:
            return self.plot_one_side(side=Side.PORT,
                                      start_idx=start_idx,
                                      end_idx=end_idx)

        if not end_idx:
            end_idx = min(len(self.sss_pings[Side.PORT]),
                          len(self.sss_pings[Side.STARBOARD]))
        left_pings = self._get_pings_from_one_side(Side.PORT, start_idx,
                                                   end_idx)
        right_pings = self._get_pings_from_one_side(Side.STARBOARD, start_idx,
                                                    end_idx)
        sss_image = np.concatenate((np.flip(left_pings, axis=1), right_pings),
                                   axis=1)
        title = f'SSS pings from {self.filename}'
        self._imshow(sss_image, start_idx, end_idx, title, figsize)
        return sss_image
