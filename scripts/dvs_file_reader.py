from enum import Enum
from dataclasses import dataclass
import utils
from typing import List, BinaryIO, Tuple
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
        self.sss_pings = []

        self._parse_file()

    def _parse_file(self):
        with open(self.filename, 'rb') as f:
            self.header = self._parse_header(f)
            while True:
                try:
                    ping = self._parse_ping(f)
                    self.sss_pings.append(ping)
                except struct.error as e:
                    pointer_pos = f.tell()
                    file_size = Path(self.filename).stat().st_size
                    print(f'Current pointer position: {pointer_pos}')
                    print(f'Total file size: {file_size}')
                    print(f'Remaining bytes: {file_size - pointer_pos}')
                    print(f'Parsing completed: {e}')
                    return

    def _parse_ping(self, fileobj: BinaryIO) -> List[SSSPing]:
        """Read one side-scan ping from the fileobj. Note that one ping
        may consists of two channels (port and starboard)."""
        lat = utils.unpack_struct(fileobj, struct_type='double')
        lon = utils.unpack_struct(fileobj, struct_type='double')
        speed = utils.unpack_struct(fileobj, struct_type='float')
        heading = utils.unpack_struct(fileobj, struct_type='float')

        ping = []
        if self.header.left:
            left_channel = utils.unpack_channel(
                fileobj, channel_size=self.header.n_samples)
            left_channel_ping = SSSPing(lat=lat,
                                        lon=lon,
                                        speed=speed,
                                        heading=heading,
                                        side=Side.PORT,
                                        ping=left_channel)
            ping.append(left_channel_ping)
        if self.header.right:
            right_channel = utils.unpack_channel(
                fileobj, channel_size=self.header.n_samples)
            right_channel_ping = SSSPing(lat=lat,
                                         lon=lon,
                                         speed=speed,
                                         heading=heading,
                                         side=Side.STARBOARD,
                                         ping=right_channel)
            ping.append(right_channel_ping)
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

    def _check_side_exists(self, side: Side) -> Tuple[bool, str]:
        """Used by method plot_one_side. Raise ValueError if the side given
        is invalid"""
        if side not in {Side.PORT, Side.STARBOARD}:
            err = f'Unknown side {side}. Please specify {Side.PORT} or {Side.STARBOARD}.'
            return False, err

        request_left_failed = side == Side.PORT and not self.header.left
        request_right_failed = side == Side.STARBOARD and not self.header.right

        if request_left_failed or request_right_failed:
            err = f'The DVSFile {self.filename} does not contain {side}'
            return False, err
        return True, ''

    def _get_ping_array_from_side(self,
                                  side: Side,
                                  start_idx: int = 0,
                                  end_idx: int = None) -> np.ndarray:
        """Returns a numpy array of pings between (start_idx, end_idx) for
        the specified side"""
        if end_idx is None:
            end_idx = len(self.sss_pings)
        print(f'start_idx = {start_idx}, end_idx = {end_idx}')

        side_pings = []
        for idx, ping in enumerate(self.sss_pings):
            if idx < start_idx or idx >= end_idx:
                continue
            for ping_part in ping:
                if ping_part.side == side:
                    side_pings.append(ping_part.ping)
        return np.array(side_pings)

    def plot_one_side(self,
                      side: Side,
                      start_idx: int = 0,
                      end_idx: int = None) -> np.ndarray:
        """Plot sss pings between (start_idx, end_idx) from the requested side
        if exists."""
        success, err = self._check_side_exists(side)
        if not success:
            raise ValueError(err)
        side_pings = self._get_ping_array_from_side(side, start_idx, end_idx)

        plt.figure()
        plt.imshow(side_pings)
        plt.title(f'SSS pings from {side} of {self.filename}')
        return side_pings

    def plot(self, start_idx: int = 0, end_idx: int = None) -> np.ndarray:
        """Plot all sss pings in the DVSFile"""
        if self.header.right and not self.header.left:
            return self.plot_one_side(side=Side.STARBOARD,
                                      start_idx=start_idx,
                                      end_idx=end_idx)
        if self.header.left and not self.header.right:
            return self.plot_one_side(side=Side.PORT,
                                      start_idx=start_idx,
                                      end_idx=end_idx)
        left_pings = self._get_ping_array_from_side(side=Side.PORT,
                                                    start_idx=start_idx,
                                                    end_idx=end_idx)
        right_pings = self._get_ping_array_from_side(side=Side.STARBOARD,
                                                     start_idx=start_idx,
                                                     end_idx=end_idx)
        sss_image = np.concatenate((np.flip(left_pings, axis=1), right_pings),
                                   axis=1)
        plt.figure()
        plt.imshow(sss_image)
        plt.title(f'SSS pings from {self.filename}')
        return sss_image
