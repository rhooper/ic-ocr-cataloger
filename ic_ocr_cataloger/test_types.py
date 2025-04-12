import unittest
from math import floor, trunc

from ic_ocr_cataloger.models import Coord, CoordSet


class TestCoord(unittest.TestCase):
    def test_coord_initialization(self):
        coord = Coord(10, 20)
        assert coord.x == 10
        assert coord.y == 20

    def test_coord_operations(self):
        coord1 = Coord(10, 20)
        coord2 = Coord(5, 5)

        assert coord1 + coord2 == Coord(15, 25)
        assert coord1 - coord2 == Coord(5, 15)
        assert coord1 * 2 == Coord(20, 40)
        assert coord1 / 2 == Coord(5.0, 10.0)

    def test_coord_string_representation(self):
        coord = Coord(10, 20)
        assert str(coord), "(10, 20)"

    def test_coord_integer_conversion(self):
        coord = Coord(10.5, 20.5)
        assert coord.integer == Coord(10, 20)

    def test_coord_all_operators(self):
        assert Coord(10, 20) + Coord(5, 5) == Coord(15, 25)
        assert Coord(10, 20) - Coord(5, 5) == Coord(5, 15)
        assert Coord(10, 20) * 2 == Coord(20, 40)
        assert Coord(10, 20) / 2 == Coord(5.0, 10.0)
        assert Coord(10, 20) // 3 == Coord(3, 6)
        assert Coord(10, 20) % 3 == Coord(1, 2)
        assert Coord(10, 20) ** 2 == Coord(100, 400)
        assert Coord(10, 20) >> 1 == Coord(5, 10)
        assert Coord(10, 20) << 1 == Coord(20, 40)
        assert Coord(10, 20) & Coord(5, 5) == Coord(0, 4)
        assert Coord(0x10, 0x20) | Coord(0x5, 0x5) == Coord(0x10 | 0x5, 0x20 | 0x5)
        assert Coord(10, 20) ^ Coord(5, 5) == Coord(10 ^ 5, 20 ^ 5)
        assert ~Coord(10, 20) == Coord(-11, -21)
        assert abs(Coord(10, 20)) == Coord(10, 20)
        assert round(Coord(10, 20)) == Coord(10, 20)
        assert -Coord(10, 20) == Coord(-10, -20)
        assert +Coord(10, 20) == Coord(10, 20)
        assert trunc(Coord(10, 20)) == Coord(10, 20)
        assert floor(Coord(10, 20)) == Coord(10, 20)
        assert ~Coord(10, 20) == Coord(-11, -21)
        assert Coord(10, 20).integer == Coord(10, 20)

    def test_coord_all_boolean_operators(self):
        assert (Coord(10, 20) and True) == True
        assert Coord(10, 20) != False
        assert (not Coord(10, 20)) != True
        assert (Coord(10, 20) == Coord(10, 20)) == True
        assert (Coord(10, 20) < Coord(15, 25)) == True
        assert (Coord(10, 20) <= Coord(15, 25)) == True
        assert bool(Coord(0, 0)) == False
        assert (Coord(10, 20) != Coord(10, 20)) == False
        assert (Coord(10, 20) > Coord(15, 25)) == False
        assert (Coord(10, 20) >= Coord(15, 25)) == False

    def test_coord_apply(self):
        assert Coord(10, 20).apply(lambda x: x * 2) == Coord(20, 40)
        assert Coord(10, 20).apply(lambda x: x + 1) == Coord(11, 21)
        assert Coord(10, 20).apply(lambda x: x - 1) == Coord(9, 19)
        assert Coord(10, 20).apply(lambda x: x / 2) == Coord(5.0, 10.0)
        assert Coord(10, 20).apply(lambda x: x // 2) == Coord(5, 10)
        assert Coord(10, 20).apply(lambda x: x % 3) == Coord(1, 2)
        assert Coord(10, 20).apply(lambda x: x**2) == Coord(100, 400)
        assert Coord(10, 20).apply(lambda x: x >> 1) == Coord(5, 10)
        assert Coord(10, 20).apply(lambda x: x << 1) == Coord(20, 40)
        assert ~Coord(10, 20) == Coord(-11, -21)

    def test_coord_chained_binary_ops(self):
        assert Coord(10, 20) + Coord(5, 5) - Coord(2, 2) * 2 / 2 == Coord(13, 23)
        assert Coord(10, 20) + Coord(5, 5) - Coord(2, 2) == Coord(13, 23)
        assert Coord(10, 20) - Coord(5, 5) + Coord(2, 2) == Coord(7, 17)
        assert Coord(10, 20) * Coord(5, 5) / Coord(2, 2) == Coord(25.0, 50.0)
        assert Coord(10, 20) / Coord(5, 5) * Coord(2, 2) == Coord(4, 8)
        assert Coord(10, 20) // Coord(5, 5) % Coord(2, 2) == Coord(0, 0)
        assert Coord(10, 20) % Coord(5, 5) // Coord(2, 2) == Coord(0, 0)
        assert Coord(10, 20) ** Coord(5, 5) >> Coord(2, 2) == Coord(25000, 800000)
        assert Coord(0x10, 0x20) >> Coord(1, 1) << Coord(2, 2) == Coord(0x20, 0x40)


class TestCoordSet(unittest.TestCase):
    def test_coord_set_initialization(self):
        coord_set = CoordSet(Coord(0, 0), Coord(1, 1), Coord(2, 2), Coord(3, 3))
        assert coord_set.top_left == Coord(0, 0)
        assert coord_set.top_right == Coord(1, 1)
        assert coord_set.bot_left == Coord(2, 2)
        assert coord_set.bot_right == Coord(3, 3)

    def setUp(self):
        self.coord_set = CoordSet(Coord(0, 4), Coord(1, 7), Coord(3, 6), Coord(2, 6))

    def test_coord_set_x(self):
        assert self.coord_set.x == (0, 1, 3, 2)

    def test_coord_set_y(self):
        assert self.coord_set.y == (4, 7, 6, 6)

    def test_coord_set_max_x(self):
        assert self.coord_set.max_x == (3, 6)

    def test_coord_set_max_y(self):
        assert self.coord_set.max_y == (1, 7)

    def test_coord_set_top_left_y(self):
        assert self.coord_set.top_left.x == 0

    def test_coord_set_top_right_y(self):
        assert self.coord_set.top_right.y == 7

    def test_coord_set_apply(self):
        assert self.coord_set.apply(lambda a: a * 2) == CoordSet(
            Coord(0, 8), Coord(2, 14), Coord(6, 12), Coord(4, 12)
        )

    def test_coord_set_string_representation(self):
        coord_set = CoordSet(Coord(0, 0), Coord(1, 1), Coord(2, 2), Coord(3, 3))
        assert (
            str(coord_set)
            == "CoordSet(top_left=Coord(x=0, y=0), top_right=Coord(x=1, y=1), bot_left=Coord(x=2, y=2), bot_right=Coord(x=3, y=3))"
        )

    def test_coord_set_integer_conversion(self):
        coord_set = CoordSet(
            Coord(10.5, 20.5), Coord(30.5, 40.5), Coord(50.5, 60.5), Coord(70.5, 80.5)
        )
        assert coord_set.integer == CoordSet(
            Coord(10, 20), Coord(30, 40), Coord(50, 60), Coord(70, 80)
        )
