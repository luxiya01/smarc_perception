from enum import Enum
from dataclasses import dataclass
import utils
from typing import List


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
                pass

    def _parse_header(self, fileobj):
        """Read version and V1_FileHeader from the file object"""
        header = DVSFileHeader()
        header.version = utils.unpack_struct(fileobj, struct_type='uint')
        header.sample_res = utils.unpack_struct(fileobj, struct_type='float')
        header.line_rate = utils.unpack_struct(fileobj, struct_type='float')
        header.n_samples = utils.unpack_struct(fileobj, struct_type='int')
        header.left = utils.unpack_struct(fileobj, struct_type='bool')
        header.right = utils.unpack_struct(fileobj, struct_type='bool')
        return header
