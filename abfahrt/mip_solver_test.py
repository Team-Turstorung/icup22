from abfahrt.mip_solver import MipSolver
from tools.test_utils import create_schedule_assert_delay


def test_simple():
    create_schedule_assert_delay(MipSolver, "examples/official/simple/input.txt", 9)


def test_kapazitaet():
    create_schedule_assert_delay(MipSolver, "examples/official/kapazität/input.txt", 0)


def test_station_capacity():
    # TODO: this is 50 in the example output because of a later detrain.
    create_schedule_assert_delay(MipSolver, "examples/official/stationCapacity/input.txt", 0)


def test_line_forth_back():
    create_schedule_assert_delay(MipSolver, "examples/official/testLineForthBack/input.txt", 0)


def test_custom_wildcard():
    create_schedule_assert_delay(MipSolver, "examples/custom/wildcard.txt", 3)


def test_custom_mip1():
    create_schedule_assert_delay(MipSolver, "examples/custom/mip_test.txt", 17)


def test_custom_mip2():
    create_schedule_assert_delay(MipSolver, "examples/custom/mip_test2.txt", 24)
