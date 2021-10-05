import networkx as nx
import matplotlib.pyplot as plt

from tools.game import GameState, Station, Line, Train, TrainPositionType, PassengerGroup, PassengerGroupPositionType


def get_mode(line: str, current_mode: str):
    if line == "[stations]":
        return "StationMode", True
    if line == "[lines]":
        return "LineMode", True
    if line == "[trains]":
        return "TrainMode", True
    if line == "[passengers]":
        return "PassengerMode", True
    return current_mode, False


def parse_text(text: str):
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
        mode, changed = get_mode(line.lower(), mode)
        if not (line.startswith('#') or line == '' or changed):
            if mode == "unknown":
                print("Missing Definiton")
            elif mode == "StationMode":
                line_list = line.split()
                if len(line_list) != 2:
                    print("Invalid station")

                station_name = line_list[0]
                capacity = int(line_list[1])
                stations[station_name] = Station(
                    name=station_name, capacity=capacity, trains=[], passenger_groups=[])
                graph.add_node(station_name)
            elif mode == "LineMode":
                line_list = line.split()
                if len(line_list) != 5:
                    print("Invalid Line")

                line_name = line_list[0]
                length = float(line_list[3])
                start = stations[line_list[1]]
                end = stations[line_list[2]]
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
                    print("Invalid Train")
                train_name = line_list[0]
                position = stations[line_list[1]
                                    ] if line_list[1] != '*' else None
                train_capacity = int(line_list[3])
                speed = float(line_list[2])
                if position is None:
                    trains[train_name] = Train(
                        name=train_name,
                        position_type=TrainPositionType.NOT_STARTED,
                        speed=speed,
                        capacity=train_capacity,
                        position=position,
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
                    stations[position.name].trains.append(trains[train_name])
            elif mode == "PassengerMode":
                line_list = line.split()
                if len(line_list) != 5:
                    print("Invalid Passenger")
                passenger_group_name = line_list[0]
                position = stations[line_list[1]]
                destination = stations[line_list[2]]
                position_type = PassengerGroupPositionType.STATION
                group_size = int(line_list[3])
                time_remaining = int(line_list[4])
                passenger_groups[passenger_group_name] = PassengerGroup(name=passenger_group_name, position=position,
                                                                        position_type=position_type, group_size=group_size,
                                                                        destination=destination, time_remaining=time_remaining)
                stations[position.name].passenger_groups.append(
                    passenger_groups[passenger_group_name])

    game_state = GameState(
        trains,
        passenger_groups,
        stations,
        train_lines)

    return game_state, graph


def make_graph(stations: dict, train_lines: dict):
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


def parse_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return parse_text(text)


# Testcode
if __name__ == "__main__":
    test_world, new_graph = parse_file("examples/official/simple/input.txt")
    print(test_world.to_dict())
    nx.draw(new_graph)
    plt.show()
