import json
import operator as op
from math import floor, trunc
from typing import Callable, NamedTuple, Self, Sequence, TypeVar, Union

from numpy import int32

OpArg = TypeVar("OpArg", bound=Union[int, float, "Coord"])


def make_binary_op(func: Callable) -> Callable[[OpArg, OpArg], "Coord"]:
    """
    Create a binary operator function for a given callable, allowing free intermixing of
    Coord(x,y) and int/float arguments.
    """

    def _apply(a: OpArg, b: OpArg) -> "Coord":
        if not isinstance(a, Coord):
            return Coord(func(a, b.x), func(a, b.y))
        elif not isinstance(b, Coord):
            return Coord(func(a.x, b), func(a.y, b))
        return Coord(func(a.x, b.x), func(a.y, b.y))

    return _apply


def make_unary_op(func: Callable) -> Callable[[OpArg], "Coord"]:
    """
    Create a unary operator function for a given callable, which applies the operator
    to both the x and y members of the Coord tuple.
    """

    def _op(a: "Coord") -> "Coord":
        return Coord(func(a.x), func(a.y))

    return _op


class BoundingBox(NamedTuple):
    confidence: float
    text: str
    position: Sequence[int]
    matched: bool = False


class Coord(NamedTuple):
    x: float | int
    y: float | int

    @property
    def integer(self) -> Self:
        return self.apply(int)

    def apply(self, func: Callable) -> Self:
        return Coord(func(self.x), func(self.y))

    __mul__ = make_binary_op(op.__mul__)
    __sub__ = make_binary_op(op.__sub__)
    __rsub__ = make_binary_op(op.__sub__)
    __add__ = make_binary_op(op.__add__)
    __radd__ = make_binary_op(op.__add__)
    __truediv__ = make_binary_op(op.__truediv__)
    __rtruediv__ = make_binary_op(op.__truediv__)
    __floordiv__ = make_binary_op(op.__floordiv__)
    __rfloordiv__ = make_binary_op(op.__floordiv__)
    __mod__ = make_binary_op(op.__mod__)
    __rmod__ = make_binary_op(op.__mod__)
    __pow__ = make_binary_op(op.__pow__)
    __rpow__ = make_binary_op(op.__pow__)
    __rshift__ = make_binary_op(op.__rshift__)
    __rrshift__ = make_binary_op(op.__rshift__)
    __lshift__ = make_binary_op(op.__lshift__)
    __rlshift__ = make_binary_op(op.__lshift__)
    __and__ = make_binary_op(op.__and__)
    __rand__ = make_binary_op(op.__and__)
    __or__ = make_binary_op(op.__or__)
    __ror__ = make_binary_op(op.__or__)
    __xor__ = make_binary_op(op.__xor__)
    __rxor__ = make_binary_op(op.__xor__)
    __abs__ = make_unary_op(op.__abs__)
    __neg__ = make_unary_op(op.__neg__)
    __pos__ = make_unary_op(op.__pos__)
    __trunc__ = make_unary_op(trunc)
    __floor__ = make_unary_op(floor)
    __int__ = make_unary_op(int)
    __invert__ = make_unary_op(op.__invert__)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __round__(self, n=0) -> Self:
        return Coord(round(self.x, n), round(self.y, n))

    def __bool__(self):
        return bool(self.x) or bool(self.y)


class CoordSet(NamedTuple):
    top_left: Coord
    top_right: Coord
    bot_left: Coord
    bot_right: Coord

    @property
    def x(self) -> tuple[float, float, float, float]:
        return self.top_left.x, self.top_right.x, self.bot_left.x, self.bot_right.x

    @property
    def y(self) -> tuple[float, float, float, float]:
        return self.top_left.y, self.top_right.y, self.bot_left.y, self.bot_right.y

    @property
    def max_x(self) -> Coord:
        return max(self, key=lambda x: (x.x, x.y))

    @property
    def max_y(self) -> Coord:
        return max(self, key=lambda y: (y.y, y.x))

    def apply(self, fn: Callable) -> Self:
        return CoordSet(*[fn(i) for i in self])

    @property
    def integer(self):
        return self.apply(lambda x: x.integer)


class JSEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle Coord and CoordSet objects.
    """

    def default(self, o):
        if isinstance(o, Coord):
            return {"x": o.x, "y": o.y}
        if isinstance(o, CoordSet):
            return {
                "top_left": o.top_left,
                "top_right": o.top_right,
                "bot_left": o.bot_left,
                "bot_right": o.bot_right,
            }
        if isinstance(o, int32):
            return int(o)
        return super().default(o)
