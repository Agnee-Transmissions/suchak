import typing
from numba.experimental import jitclass as _jitclass
from numba.experimental.jitclass.base import JitClassType


def jitclass(cls: typing.Type) -> typing.Type:
    try:
        spec = dict(cls.__annotations__)
        del cls.__annotations__
    except AttributeError:
        spec = {}

    values = {}
    for name, typ in spec.items():
        try:
            value = getattr(cls, name)
        except AttributeError:
            pass
        else:
            values[name] = value
            delattr(cls, name)

        if isinstance(typ, JitClassType):
            spec[name] = typ.class_type.instance_type

    cls = _jitclass(spec)(cls)

    def default():
        self = cls()

        for name, value in values.items():
            setattr(self, name, value)

        return self

    cls.default = default

    return cls
