import struct
from collections import namedtuple

STRUCT = {
    'int': {
        'size': 4,
        'fmt': 'i'
    },
    'uint': {
        'size': 4,
        'fmt': 'I'
    },
    'bool': {
        'size': 1,
        'fmt': '?'
    },
    'float': {
        'size': 4,
        'fmt': 'f'
    },
    'double': {
        'size': 8,
        'fmt': 'd'
    }
}


def unpack_struct(fileobj, struct_type):
    """Given a file object and a struct type, read the corresponding number of bytes in
    standard size, unpack the resulting buffer according to the correct format and return
    the unpacked object. Assume native byte order."""
    info = STRUCT[struct_type]
    buff = fileobj.read(info['size'])
    return struct.unpack(info['fmt'], buff)[0]


def unpack_channel(fileobj, channel_size):
    """Given a file object, read channel_size number of bytes from the fileobj, unpack
    the resulting buffer as a char array with correct size. Assume native byte order.
    Returns a List(int) of length = channel_size."""
    fmt = f'{channel_size}B'
    buff = fileobj.read(channel_size)
    return list(struct.unpack(fmt, buff))
