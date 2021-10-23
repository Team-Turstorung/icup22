from tools.file_parser import parse_input_file


def create_schedule_assert_delay(solver, input_file, delay):
    network_state, network_graph = parse_input_file(input_file)
    solver = solver(network_state, network_graph)
    mip_schedule = solver.schedule()
    network_state.apply_all(mip_schedule)
    assert network_state.is_valid()
    assert network_state.is_finished()
    assert network_state.total_delay() == delay
