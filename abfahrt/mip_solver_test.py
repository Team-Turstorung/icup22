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
