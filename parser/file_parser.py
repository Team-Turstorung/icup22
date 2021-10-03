import networkx as nx
import matplotlib.pyplot as plt


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
    passengers = {}
    stations = {}
    train_lines = {}
    lines = text.splitlines()
    mode = "unknown"
    for line in lines:
        # Remove unneccesseary whitespace
        line = line.strip()
        mode, changed = get_mode(line.lower(), mode)
        if not(line.startswith('#') or line == '' or changed):
            if mode == "unknown":
                print("Missing Definiton")
            elif mode == "StationMode":
                line_list = line.split()
                if len(line_list) != 2:
                    print("Invalid station")
                stations[line_list[0]] = {"capacity": int(line_list[1])}
            elif mode == "LineMode":
                line_list = line.split()
                if len(line_list) != 5:
                    print("Invalid Line")
                train_lines[line_list[0]] = {"start": line_list[1], "end": line_list[2], "length": float(
                    line_list[3]), "capacity": int(line_list[4])}
            elif mode == "TrainMode":
                line_list = line.split()
                if len(line_list) != 4:
                    print("Invalid Train")
                trains[line_list[0]] = {"position": line_list[1], "speed": float(
                    line_list[2]), "capacity": int(line_list[3])}
            elif mode == "PassengerMode":
                line_list = line.split()
                if len(line_list) != 5:
                    print("Invalid Passenger")
                passengers[line_list[0]] = {"start": line_list[1],
                                            "destination": line_list[2],
                                            "size": int(line_list[3]),
                                            "time": int(line_list[4])}
    return make_graph(stations, passengers, train_lines, trains)


def make_graph(stations: dict, passengers: dict,
               train_lines: dict, trains: dict):
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
    nx.draw(test_world)
    plt.show()
