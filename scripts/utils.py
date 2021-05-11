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
    the unpacked object"""
    info = STRUCT[struct_type]
    buff = fileobj.read(info['size'])
    return struct.unpack(info['fmt'], buff)[0]
