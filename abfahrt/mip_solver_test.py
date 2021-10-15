from abfahrt.mip_solver import MipSolver
from tools.file_parser import parse_input_file


def create_schedule_assert_delay(input_file, delay):
    solver = MipSolver()
    network_state, network_graph = parse_input_file(input_file)
    mip_schedule = solver.schedule(network_state, network_graph)
    network_state.apply_all(mip_schedule)
    assert network_state.is_valid()
    assert network_state.is_finished()
    assert network_state.total_delay() == delay


def test_simple():
    create_schedule_assert_delay("examples/official/simple/input.txt", 9)


def test_kapazitaet():
    create_schedule_assert_delay("examples/official/kapazität/input.txt", 0)


def test_station_capacity():
    # TODO: this is 50 in the example output because of a later detrain.
    create_schedule_assert_delay("examples/official/stationCapacity/input.txt", 0)


def test_line_forth_back():
    create_schedule_assert_delay("examples/official/testLineForthBack/input.txt", 0)


def test_custom_wildcard():
    create_schedule_assert_delay("examples/custom/wildcard/input.txt", 3)
