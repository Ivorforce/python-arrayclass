import dataclasses
import functools
import typing
import numpy as np


def from_array(cls, array):
    array = np.asarray(array, dtype=cls.__VALUES_TYPE__)
    if cls.__VALUES_LEN__ != len(array):
        raise RuntimeError(
            f"cannot construct arrayclass '{cls}' of length {cls.__VALUES_LEN__} from array of length {len(array)}"
        )

    return cls(*np.split(array, cls.__VALUES_OFFSETS__[1:]))


def to_array(object, *args, **kwargs):
    return np.array(object.values, *args, **kwargs)


def get_len(object):
    return len(object.values)


def getitem(object, item):
    return object.values[item]


def setitem(object, item, value):
    object.values[item] = value


@functools.wraps(dataclasses.dataclass)
def arrayclass(
    cls=None,
    /,
    dtype=None,
    **kwargs
):
    """Calls dataclasses.dataclass. Also adds the following:
    object.values
    object.__array__
    object.__len__
    object.__getitem__
    object.__setitem__

    class.__VALUES_OFFSETS__
    class.__VALUES_LEN__
    """

    def wrap(cls):
        return _process_class(cls, dtype=dtype, **kwargs)

    # See if we're being called as @arrayclass or @arrayclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called without parens.
    return wrap(cls)


def _process_class(cls, dtype=None, **kwargs):
    # Copied from dataclasses._process_class, it's not that bad to redo this
    cls_annotations = cls.__dict__.get("__annotations__", {})
    cls_fields = [
        dataclasses._get_field(cls, name, type)
        for name, type in cls_annotations.items()
    ]

    # Go through all the fields and prepare getters / setters.
    value_count = 0
    properties = dict()
    values_offsets = []
    types = []

    for f in cls_fields:
        type_args = typing.get_args(f.type)
        field_length = len(type_args)
        if field_length == 0:
            # Single value
            index = value_count
            field_length = 1
            types.append(f.type)
        else:
            # Multi-Value
            index = slice(value_count, value_count + field_length)
            types.extend(type_args)

        # TODO index=index is a dirty hack; it would be better to construct the function from code
        # but eh, that takes effort, right?
        def get(self, *, index=index):
            return self.values[index]

        def set(self, value, *, index=index):
            self.values[index] = value

        properties[f.name] = property(fget=get, fset=set)

        values_offsets.append(value_count)
        value_count += field_length

    if dtype is None:
        dtype = np.find_common_type([], types)

    # Make it a dataclass!
    cls = dataclasses.dataclass(cls, **kwargs)

    # Override the init
    @functools.wraps(cls.__init__)
    def init(self, *args, **kwargs):
        self.values = np.empty(value_count, dtype=dtype)
        self.__dataclass_init__(*args, **kwargs)

    cls.__dataclass_init__ = cls.__init__
    cls.__init__ = init

    # Add the created properties.
    # There's no references besides __init__ that values should be in __dict__ - so we can just make them properties.
    # This needs to happen after dataclasses.dataclass() because that one queries the assignments (field() or defaults)
    for name, prop in properties.items():
        setattr(cls, name, prop)

    # Make us accessible like an array, so things like tuple(obj) and np.array(obj) work.
    cls.__array__ = to_array
    cls.__len__ = get_len
    cls.__getitem__ = getitem
    cls.__setitem__ = setitem

    # Used in from_array()
    cls.__VALUES_TYPE__ = dtype
    cls.__VALUES_OFFSETS__ = values_offsets
    cls.__VALUES_LEN__ = value_count

    return cls
