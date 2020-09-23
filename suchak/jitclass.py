import inspect
import typing

import numba
from numba import njit
from numba.experimental import jitclass as _jitclass
from numba.experimental.jitclass.base import JitClassType

_annotations = {}


def jitclass(cls: typing.Type) -> typing.Type:
    if numba.config.DISABLE_JIT:
        return cls

    # print()
    # print(cls)

    try:
        annos = cls.__annotations__
    except AttributeError:
        spec = {}
    else:
        spec = dict(annos)
        del cls.__annotations__
        _annotations[cls.__qualname__] = spec

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

    if values:
        sep = "\n "

        body = ""
        init_locals = {}

        for name, value in values.items():
            newname = f"__init__local__{name}"
            body += sep + f"self.{name} = {newname}"
            init_locals[newname] = value

        if isinstance(cls.__init__, type(object.__init__)):
            cls__init__sig = "(self)"
        else:
            init_locals["__cls__init__"] = njit(cls.__init__)

            signature = inspect.signature(cls.__init__)
            param_names = ", ".join(signature.parameters)
            cls__init__sig = f"({param_names})"

            body += sep + f"__cls__init__{cls__init__sig}"

        code = f"def __jitcls__init__{cls__init__sig}:{body}"

        # print(code)
        # print()

        exec(code, init_locals)

        cls.__init__ = init_locals["__jitcls__init__"]

    return _jitclass(spec)(cls)
