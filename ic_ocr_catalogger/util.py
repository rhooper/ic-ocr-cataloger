import collections
import json
import operator as op
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from itertools import pairwise
from math import trunc, floor, atan, degrees
from typing import NamedTuple, Sequence, Self, TypeVar, Union

from PIL import ImageFont
from numpy import int32

BIG_FONT = ImageFont.truetype("NotoSansNerdFont-Regular", 40)
OpArg = TypeVar("OpArg", bound=Union[int, float, "Coord"])


def count_matches(recent):
    return sum(len(f) for f in recent)


class FlashMessage(NamedTuple):
    expire: datetime
    text: str
    color: tuple[int, int, int] = (128, 128, 128)
    font: ImageFont = BIG_FONT


class BoundingBox(NamedTuple):
    confidence: float
    text: str
    position: Sequence[int]
    matched: bool = False


def aggregate(frame):
    return Counter({f.part.part_no: v for f, v in frame.items()})


def _make_binary_op(func: Callable) -> Callable[[OpArg, OpArg], "Coord"]:
    def _apply(a: OpArg, b: OpArg) -> "Coord":
        if not isinstance(a, Coord):
            return Coord(func(a, b.x), func(a, b.y))
        elif not isinstance(b, Coord):
            return Coord(func(a.x, b), func(a.y, b))
        return Coord(func(a.x, b.x), func(a.y, b.y))

    return _apply


def _make_unary_op(func: Callable) -> Callable[[OpArg], "Coord"]:
    def _op(a: "Coord") -> "Coord":
        return Coord(func(a.x), func(a.y))

    return _op


class Coord(NamedTuple):
    x: float | int
    y: float | int

    @property
    def integer(self) -> Self:
        return self.apply(int)

    def apply(self, func: Callable) -> Self:
        return Coord(func(self.x), func(self.y))

    __mul__ = _make_binary_op(op.__mul__)
    __sub__ = _make_binary_op(op.__sub__)
    __add__ = _make_binary_op(op.__add__)
    __truediv__ = _make_binary_op(op.__truediv__)
    __floordiv__ = _make_binary_op(op.__floordiv__)
    __mod__ = _make_binary_op(op.__mod__)
    __pow__ = _make_binary_op(op.__pow__)
    __rshift__ = _make_binary_op(op.__rshift__)
    __lshift__ = _make_binary_op(op.__lshift__)
    __and__ = _make_binary_op(op.__and__)
    __or__ = _make_binary_op(op.__or__)
    __xor__ = _make_binary_op(op.__xor__)
    __round__ = _make_binary_op(round)
    __abs__ = _make_unary_op(op.__abs__)
    __neg__ = _make_unary_op(op.__neg__)
    __pos__ = _make_unary_op(op.__pos__)
    __float__ = _make_unary_op(float)
    __trunc__ = _make_unary_op(trunc)
    __floor__ = _make_unary_op(floor)
    __invert__ = _make_unary_op(op.__invert__)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


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


class TextElement(NamedTuple):
    coords: CoordSet
    shape: Coord
    text: str
    confidence: float

    def int(self):
        return TextElement(
            self.coords,
            self.shape,
            self.text,
            int(self.confidence),
        )


def layout_row(buf):
    if not buf:
        return

    for line in buf:
        if not line:
            yield ""
            continue

        pre_pad = " " * int(line[0][0] / 2)
        if len(line) == 1:
            yield pre_pad + line[0][1]
            continue
        out = [pre_pad]
        size = len(pre_pad)
        for (left_col, left_text), (right_col, right_text) in pairwise(
            sorted(line, key=lambda i: i[0])
        ):
            size += len(left_text) + 1
            out.append(left_text)
            if right_col > size:
                pad = " " * int(right_col - size)
                out.append(pad)
        # When done, append the right text
        out.append(line[-1][1])
        yield " ".join(out)


def find_angle(points: CoordSet):
    a = points.top_left
    b = points.bot_left
    c = points.bot_right
    m1 = slope(b, a)
    m2 = slope(b, c)
    calc_angle = (m2 - m1) / (1 + m1 * m2)
    rad_angle = atan(calc_angle)
    degree_angle = round(degrees(rad_angle))
    if degree_angle < 0:
        degree_angle = 180 + degree_angle
    return degree_angle


# check for skew or rotation
# compute angle
def slope(p1, p2):
    try:
        return (p2.y - p1.y) / (p2.x - p1.x)
    except ZeroDivisionError:
        return 0


class AutoGrowList(collections.UserList):

    def __init__(self, default: type = str):
        self.default = default
        super().__init__()

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        end = i if not isinstance(i, slice) else (i[1] or len(i))
        if end >= len(self):
            self.extend([self.default() for _ in range(end - len(self) + 1)])
        if isinstance(i, slice):
            return super().__getitem__(i)
        return super().__getitem__(i)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            return super().__setitem__(key, value)
        if key >= len(self):
            self.extend([self.default() for _ in range(key - len(self) + 1)])
        super().__setitem__(key, value)


class JSEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Coord):
            return {"x": o.x, "y": o.y}
        if isinstance(o, TextElement):
            return {
                "coords": o.coords,
                "shape": o.shape,
                "text": o.text,
                "confidence": o.confidence,
            }
        if isinstance(o, CoordSet):
            return {
                "top_left": o.top_left,
                "top_right": o.top_right,
                "bot_left": o.bot_left,
                "bot_right": o.bot_right,
            }
        if isinstance(o, AutoGrowList):
            return list(o)
        if isinstance(o, int32):
            return int(o)
        return super().default(o)


def reformat_json(param):
    json_struct = json.loads(param)

    yield "{\n"
    yield f"    text: {json.dumps(json_struct['text'])},\n"
    yield f"    lines: [\n"
    for line in json_struct["lines"]:
        yield f"        {json.dumps(sorted(line))},\n"
    yield f"    ],\n"
    yield f"    raw: [\n"
    for item in sorted(json_struct["raw"], key=lambda x: x[0][0][1]):
        yield f"        {json.dumps(item)},\n"
    yield f"    ]\n"
    yield "}\n"
