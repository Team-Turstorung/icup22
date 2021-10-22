from abfahrt.simple_solution_multiple_trains import SimplesSolverMultipleTrains
from tools.test_utils import create_schedule_assert_delay


def test_simple():
    # TODO optimally, this is 9
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/official/simple/input.txt", 24)


def test_kapazitaet():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/official/kapazität/input.txt", 0)


def test_station_capacity():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/official/stationCapacity/input.txt", 0)


def test_line_forth_back():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/official/testLineForthBack/input.txt", 0)

# TODO implement wildcard trains that are not used
# def test_custom_wildcard():
#     create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/custom/wildcard/input.txt", 3)


def test_custom_mip1():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/custom/mip_test.txt", 17)


def test_custom_mip2():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/custom/mip_test2.txt", 24)


def test_custom_mip3():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/custom/mip_test3.txt", 371)


def test_custom_swap():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/custom/swap.txt", 4)


def test_custom_blockchain():
    create_schedule_assert_delay(SimplesSolverMultipleTrains, "examples/custom/blockchain.txt", 4)
