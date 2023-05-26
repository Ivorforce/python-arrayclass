import dataclasses
import functools
import typing
import numpy as np


def from_array(cls, array):
    if cls.__VALUES_LEN__ != len(array):
        raise RuntimeError(
            f"cannot construct arrayclass '{cls}' of length {cls.__VALUES_LEN__} from array of length {len(array)}"
        )

    return cls(*np.split(array, cls.__VALUES_OFFSETS__[1:]))


def to_array(object):
    return np.copy(object.values)


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
        return _process_class(cls, **kwargs)

    # See if we're being called as @arrayclass or @arrayclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called without parens.
    return wrap(cls)


def _process_class(cls, **kwargs):
    # Copied from dataclasses._process_class, it's not that bad to redo this
    cls_annotations = cls.__dict__.get("__annotations__", {})
    cls_fields = [
        dataclasses._get_field(cls, name, type)
        for name, type in cls_annotations.items()
    ]

    value_count = 0
    properties = dict()
    values_offsets = []

    # This will be called when the first setter is called in __init__.
    def create_values_field(self):
        self.values = np.empty(value_count)

    # Go through all the fields and prepare getters / setters.
    for f in cls_fields:
        field_length = len(typing.get_args(f.type))
        if field_length == 0:
            # Single value
            index = value_count
            field_length = 1
        else:
            # Multi-Value
            index = slice(value_count, value_count + field_length)

        # index=index is a dirty hack; it would be better to construct the function from code
        # but eh, that takes effort, right?
        def get(self, *, index=index):
            return self.values[index]

        def set(self, value, *, index=index):
            if not hasattr(self, "values"):
                create_values_field(self)
            self.values[index] = value

        properties[f.name] = property(fget=get, fset=set)

        values_offsets.append(value_count)
        value_count += field_length

    # cls.values will be treated as if added by the user, resolving to a field
    cls = dataclasses._process_class(cls, **kwargs)

    # Add the created properties.
    # Needs to happen after dataclasses.dataclass() because that one queries the assignments (field() or defaults)
    for name, prop in properties.items():
        setattr(cls, name, prop)
        pass

    # Make us accessible like an array, so things like tuple(obj) and np.array(obj) work.
    cls.__array__ = to_array
    cls.__len__ = get_len
    cls.__getitem__ = getitem
    cls.__setitem__ = setitem

    # Used in from_array()
    cls.__VALUES_OFFSETS__ = values_offsets
    cls.__VALUES_LEN__ = value_count

    return cls
