from tools.file_parser import parse_input_file, parse_output_file


def test_simple():
    test_world, _ = parse_input_file("examples/official/simple/input.txt")
    schedule = parse_output_file("examples/official/simple/output.txt")
    test_world.apply_all(schedule)
    assert test_world.is_finished()


def test_cap():
    test_world, _ = parse_input_file("examples/official/kapazität/input.txt")
    schedule = parse_output_file("examples/official/kapazität/output.txt")
    test_world.apply_all(schedule)
    assert test_world.is_finished()


def test_station_cap():
    test_world, _ = parse_input_file("examples/official/stationCapacity/input.txt")
    schedule = parse_output_file("examples/official/stationCapacity/output.txt")
    test_world.apply_all(schedule)
    assert test_world.is_finished()


def test_line_forth_back():
    test_world, _ = parse_input_file("examples/official/testLineForthBack/input.txt")
    schedule = parse_output_file("examples/official/testLineForthBack/output.txt")
    test_world.apply_all(schedule)
    assert test_world.is_finished()


def test_unused_wildcard_train():
    test_world, _ = parse_input_file("examples/official/unusedWildcardTrain/input.txt")
    schedule = parse_output_file("examples/official/unusedWildcardTrain/output.txt")
    test_world.apply_all(schedule)
    assert test_world.is_finished()
