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
    # TODO: can we safely assume that all stations come before all lines and so on? (otherwise, this might fail)
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
                stations[line_list[0]] = Station(line_list[0], int(line_list[1]), [], [])
                graph.add_node(line_list[0], capacity=int(line_list[1]))
            elif mode == "LineMode":
                line_list = line.split()
                if len(line_list) != 5:
                    print("Invalid Line")
                train_lines[line_list[0]] = Line(line_list[0], float(line_list[3]), stations[line_list[1]],
                                                 stations[line_list[2]], int(line_list[4]), [])
                graph.add_edge(line_list[1], line_list[2], id=line_list[0], length=float(line_list[3]), capacity=int(line_list[4]))
            elif mode == "TrainMode":
                line_list = line.split()
                if len(line_list) != 4:
                    print("Invalid Train")
                if line_list[1] == '*':
                    trains[line_list[0]] = Train(line_list[0], None, TrainPositionType.NOT_STARTED, float(
                        line_list[2]), int(line_list[3]), [])
                else:
                    trains[line_list[0]] = Train(line_list[0], stations[line_list[1]], TrainPositionType.STATION, float(
                        line_list[2]), int(line_list[3]), [])
                    stations[line_list[1]].trains.append(trains[line_list[0]])
            elif mode == "PassengerMode":
                line_list = line.split()
                if len(line_list) != 5:
                    print("Invalid Passenger")
                passenger_groups[line_list[0]] = PassengerGroup(line_list[0], stations[line_list[1]],
                                                                PassengerGroupPositionType.STATION, int(line_list[3]),
                                                                stations[line_list[2]], int(line_list[4]))
                stations[line_list[1]].passenger_groups.append(passenger_groups[line_list[0]])

    game_state = GameState(graph, trains, passenger_groups, stations, train_lines)

    return game_state


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
    test_world = parse_file("newInput.txt")
    nx.draw(test_world.graph)
    plt.show()
