import networkx as nx
import matplotlib.pyplot as plt


def get_mode(line, current_mode):
        if line == "[stations]":
            return "StationMode", True
        elif line == "[lines]":
            return  "LineMode", True
        elif line == "[trains]":
            return  "TrainMode", True
        elif line == "[passengers]":
            return "PassengerMode", True
        else:
            return current_mode, False

def parse_text(text):
    trains = {}
    passengers = {}
    stations = {}
    train_lines = {}
    lines = text.splitlines()
    mode = "unknown"
    for line in lines:
        #Remove unneccesseary whitespace
        line = line.strip()
        mode, changed = get_mode(line.lower(), mode)
        if not(line.startswith('#') or line == '' or changed):
            if mode == "unknown":
                print("Missing Definiton")
            elif mode == "StationMode":
                lineList = line.split()
                if len(lineList) != 2:
                    print("Invalid station")
                stations[lineList[0]] = {"capacity": int(lineList[1])}
            elif mode == "LineMode":
                lineList = line.split()   
                if len(lineList) != 5:
                    print("Invalid Line")
                train_lines[lineList[0]] = {"start": lineList[1], "end": lineList[2], "length": float(lineList[3]), "capacity": int(lineList[4])}
            elif mode == "TrainMode":
                lineList = line.split()
                if len(lineList) != 4:
                    print("Invalid Train")
                trains[lineList[0]] = {"position":lineList[1], "speed":float(lineList[2]), "capacity":int(lineList[3])}
            elif mode == "PassengerMode":
                lineList = line.split()
                if len(lineList) != 5:
                    print("Invalid Passenger")
                passengers[lineList[0]] = {"start":lineList[1], "destination":lineList[2], "size":int(lineList[3]), "time": int(lineList[4])}
    return make_graph(stations, passengers, train_lines, trains)

def make_graph(stations, passengers, train_lines, trains):
    world = nx.Graph()
    for station_id in stations.keys():
        world.add_node(station_id, capacity=stations[station_id]["capacity"])

    for line_id  in train_lines.keys():
        world.add_edge(train_lines[line_id]["start"], train_lines[line_id]["end"], length=train_lines[line_id]["length"], id=line_id, capacity=train_lines[line_id]["capacity"])
    return world

def parse_file(file_path):
    with open(file_path) as file:
        text = file.read()
        return parse_text(text)

# Testcode
if __name__ == "__main__":
    world = parse_file("newInput.txt")
    nx.draw(world)
    plt.show()
