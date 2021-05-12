from enum import Enum
from dataclasses import dataclass
import utils
from typing import List
import struct
from pathlib import Path


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
    def __init__(self, filename):
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
                    for ping in self._parse_ping(f):
                        self.sss_pings.append(ping)
                except struct.error as e:
                    pointer_pos = f.tell()
                    file_size = Path(self.filename).stat().st_size
                    print(f'Current pointer position: {pointer_pos}')
                    print(f'Total file size: {file_size}')
                    print(f'Remaining bytes: {file_size - pointer_pos}')
                    print(f'Parsing completed: {e}')
                    return

    def _parse_ping(self, fileobj):
        """Read one side-scan ping from the fileobj. Note that one ping
        may consists of two channels (port and starboard)."""
        lat = utils.unpack_struct(fileobj, struct_type='double')
        lon = utils.unpack_struct(fileobj, struct_type='double')
        speed = utils.unpack_struct(fileobj, struct_type='float')
        heading = utils.unpack_struct(fileobj, struct_type='float')
        if self.header.left:
            left_channel = utils.unpack_channel(
                fileobj, channel_size=self.header.n_samples)
            yield SSSPing(lat=lat,
                          lon=lon,
                          speed=speed,
                          heading=heading,
                          side=Side.PORT,
                          ping=left_channel)
        if self.header.right:
            right_channel = utils.unpack_channel(
                fileobj, channel_size=self.header.n_samples)
            yield SSSPing(lat=lat,
                          lon=lon,
                          speed=speed,
                          heading=heading,
                          side=Side.STARBOARD,
                          ping=right_channel)

    def _parse_header(self, fileobj):
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
