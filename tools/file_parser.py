from collections import defaultdict
import sys

import networkx as nx

from abfahrt.types import NetworkState, Station, Line, Train, TrainPositionType, PassengerGroup, PassengerGroupPositionType, \
    Schedule, RoundAction


def get_input_mode(line: str, current_mode: str) -> (str, bool):
    if line == "[stations]":
        return "StationMode", True
    if line == "[lines]":
        return "LineMode", True
    if line == "[trains]":
        return "TrainMode", True
    if line == "[passengers]":
        return "PassengerMode", True
    return current_mode, False


def parse_input_text(text: str) -> [NetworkState, nx.Graph]:
    trains = {}
    passenger_groups = {}
    stations = {}
    train_lines = {}
    lines = text.splitlines()
    mode = "unknown"
    graph = nx.Graph()
    # TODO: can we safely assume that all stations come before all lines and
    # so on? (otherwise, this might fail)
    for line in lines:
        # Remove unnecessary whitespace
        line = line.strip()
        mode, changed = get_input_mode(line.lower(), mode)
        if not (line.startswith('#') or line == '' or changed):
            if mode == "unknown":
                raise Exception(f"missing definition while parsing line \"{line}\"")
            elif mode == "StationMode":
                line_list = line.split()
                if len(line_list) != 2:
                    raise Exception("invalid station while parsing line \"{line}\"")

                station_name = line_list[0]
                capacity = int(line_list[1])
                stations[station_name] = Station(
                    name=station_name, capacity=capacity, trains=[], passenger_groups=[])
                graph.add_node(station_name)
            elif mode == "LineMode":
                line_list = line.split()
                if len(line_list) != 5:
                    raise Exception("invalid line while parsing line \"{line}\"")

                line_name = line_list[0]
                length = float(line_list[3])
                start = line_list[1]
                end = line_list[2]
                capacity = int(line_list[4])
                train_lines[line_name] = Line(
                    name=line_name,
                    length=length,
                    start=start,
                    end=end,
                    capacity=capacity,
                    trains=[])
                graph.add_edge(
                    line_list[1],
                    line_list[2],
                    weight=length,
                    name=line_name)
            elif mode == "TrainMode":
                line_list = line.split()
                if len(line_list) != 4:
                    raise Exception("invalid train while parsing line \"{line}\"")
                train_name = line_list[0]
                position = line_list[1]
                train_capacity = int(line_list[3])
                speed = float(line_list[2])
                if position == '*':
                    trains[train_name] = Train(
                        name=train_name,
                        position_type=TrainPositionType.NOT_STARTED,
                        speed=speed,
                        capacity=train_capacity,
                        position=None,
                        line_progress=0,
                        next_station=None,
                        passenger_groups=[])
                else:
                    trains[train_name] = Train(
                        name=train_name,
                        position_type=TrainPositionType.STATION,
                        speed=speed,
                        capacity=train_capacity,
                        position=position,
                        line_progress=0,
                        next_station=None,
                        passenger_groups=[])
                    stations[position].trains.append(train_name)
            elif mode == "PassengerMode":
                line_list = line.split()
                if len(line_list) != 5:
                    raise Exception("invalid passenger group while parsing line \"{line}\"")
                passenger_group_name = line_list[0]
                position = line_list[1]
                destination = line_list[2]
                position_type = PassengerGroupPositionType.STATION
                group_size = int(line_list[3])
                time_remaining = int(line_list[4])
                passenger_groups[passenger_group_name] = PassengerGroup(name=passenger_group_name, position=position,
                                                                        position_type=position_type, group_size=group_size,
                                                                        destination=destination, time_remaining=time_remaining)
                stations[position].passenger_groups.append(passenger_group_name)

    game_state = NetworkState(
        trains,
        passenger_groups,
        stations,
        train_lines)

    return game_state, graph


def get_output_mode(line: str, current_mode: str, current_mode_id: str) -> (str, bool, str):
    if line.startswith("[Train:") and line.endswith("]"):
        return "TrainMode", True, line[7:-1]
    if line.startswith("[Passenger") and line.endswith("]"):
        return "PassengerMode", True, line[11:-1]
    return current_mode, False, current_mode_id


def parse_output_text(text: str) -> Schedule:
    sched = Schedule(actions=defaultdict(RoundAction))
    mode = "unknown"
    mode_id = ""
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        mode, changed, mode_id = get_output_mode(line, mode, mode_id)
        if not (line.startswith('#') or line == '' or changed):
            line_list = line.split()
            if mode == "TrainMode":
                if len(line_list) != 3:
                    raise Exception(f"train action '{line}' must have 3 arguments")
                if line_list[1] == "Start":
                    if line_list[0] != "0":
                        raise Exception("trains must be started in round 0")
                    sched.actions[0].train_starts[mode_id] = line_list[2]
                elif line_list[1] == "Depart":
                    sched.actions[int(line_list[0])].train_departs[mode_id] = line_list[2]
                else:
                    raise Exception(f"invalid line '{line}'")
            elif mode == "PassengerMode":
                if len(line_list) < 2:
                    raise Exception(f"passenger action '{line}' must have 2+ arguments")
                if line_list[1] == "Detrain":
                    sched.actions[int(line_list[0])].passenger_detrains.append(mode_id)
                elif line_list[1] == "Board":
                    if len(line_list) != 3:
                        raise Exception(f"boarding action '{line}' must have 3 arguments")
                    sched.actions[int(line_list[0])].passenger_boards[mode_id] = line_list[2]
                else:
                    raise Exception(f"invalid line '{line}'")
            else:
                raise Exception(f"mode unclear for '{line}'")
    return sched


def make_graph(stations: dict, train_lines: dict) -> nx.Graph:
    world = nx.Graph()
    for station_id in stations.keys():
        world.add_node(station_id, capacity=stations[station_id]["capacity"])

    for line_id in train_lines.keys():
        world.add_edge(
            train_lines[line_id]["start"],
            train_lines[line_id]["end"],
            length=train_lines[line_id]["length"],
            id=line_id,
            capacity=train_lines[line_id]["capacity"])
    return world


def parse_input_file(file_path: str) -> (NetworkState, nx.Graph):
    if file_path == '-':
        text = sys.stdin.read()
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    return parse_input_text(text)


def parse_output_file(file_path: str) -> Schedule:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return parse_output_text(text)
