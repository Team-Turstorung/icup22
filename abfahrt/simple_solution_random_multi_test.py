from abfahrt.simple_solution_random_multi import SimpleSolverMultipleTrainsRandom
from tools.test_utils import create_schedule_assert_delay


def test_simple():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/official/simple/input.txt", 9)


def test_kapazitaet():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/official/kapazität/input.txt", 0)


def test_station_capacity():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/official/stationCapacity/input.txt", 0)


def test_line_forth_back():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/official/testLineForthBack/input.txt", 0)


def test_custom_wildcard():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/official/unusedWildcardTrain/input.txt", 68)

def test_custom_mip1():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/custom/mip_test.txt", 17)


def test_custom_mip2():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/custom/mip_test2.txt", 24)


def test_custom_mip3():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/custom/mip_test3.txt", 371)


def test_custom_swap():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/custom/swap.txt", 4)


def test_custom_blockchain():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/custom/blockchain.txt", 4)


def test_custom_line_limit():
    create_schedule_assert_delay(SimpleSolverMultipleTrainsRandom, "examples/custom/line_limit_conflict.txt", 7)
