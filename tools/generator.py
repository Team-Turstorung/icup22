import math
import random
import argparse

import networkx as nx
import matplotlib.pyplot as plt

from tools.game import GameState, Station, Line, Train, TrainPositionType, PassengerGroup, PassengerGroupPositionType


# adapted from https://stackoverflow.com/a/14618505


def random_connected_graph(nodes, density=0):
    if density >= 1:
        return nx.complete_graph(nodes)
    # Create two partitions, S and T. Initially store all nodes in S.
    set_1, set_2 = set(nodes), set()
    edge_counter = 1

    # Pick a random node, and mark it as visited and the current node.
    current_node = random.choice(nodes)
    set_1.remove(current_node)
    set_2.add(current_node)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    # Create a random connected graph.
    while set_1:
        # Randomly pick the next node from the neighbors of the current node.
        # As we are generating a connected graph, we assume a complete graph.
        neighbor_node = random.choice(nodes)
        # If the new node hasn't been visited, add the edge from current to
        # new.
        if neighbor_node not in set_2:
            graph.add_edge(current_node, neighbor_node,
                           name='L' + str(edge_counter))
            set_1.remove(neighbor_node)
            set_2.add(neighbor_node)
            edge_counter += 1
        # Set the new node as the current node.
        current_node = neighbor_node

    missing_edges = (graph.number_of_nodes(
    ) * (graph.number_of_nodes() - 1) * density) // 2 - graph.number_of_edges()
    while missing_edges > 0:
        n_1, n_2 = random.choice(nodes), random.choice(nodes)
        if not graph.has_edge(n_1, n_2) and not graph.has_edge(
            n_2, n_1) and n_1 != n_2:
            graph.add_edge(n_1, n_2, name='L' + str(edge_counter))
            edge_counter += 1
            missing_edges -= 1

    return graph


def generate_game_state(**kwargs) -> [GameState, nx.Graph]:
    num_stations = kwargs.get('num_stations', random.randint(5, 10))
    num_trains = kwargs.get('num_trains', random.randint(
        max(1, math.floor(num_stations * 0.5)), math.ceil(num_stations * 1.2)))
    num_passengers = kwargs.get('num_passengers', random.randint(1, math.floor(num_stations * 0.5)))
    density = kwargs.get('density', 0)

    min_line_capacity = kwargs.get('min_line_capacity', 1)
    max_line_capacity = kwargs.get('max_line_capacity', 5)
    min_line_length = kwargs.get('min_line_length', 1)
    max_line_length = kwargs.get('max_line_length', 10)
    min_train_speed = kwargs.get('min_train_speed', 1)
    max_train_speed = kwargs.get('max_train_speed', 10)
    min_station_capacity = kwargs.get('min_station_capacity', 1)
    max_station_capacity = kwargs.get('max_station_capacity', 5)
    min_train_capacity = kwargs.get('min_train_capacity', 1)
    max_train_capacity = kwargs.get('max_train_capacity', 60)
    min_group_size = kwargs.get('min_group_size', 1)
    max_group_size = kwargs.get('min_group_size', 15)
    min_time = kwargs.get('min_time', 1)
    max_time = kwargs.get('min_time', 10)

    # some sanity checks
    if min_train_capacity < max_group_size:
        print(
            "WARNING: passenger groups might be too large to be transported by any train")  # TODO: take this into account when assigning train capacities / group sizes?
    if num_trains > num_stations * min_station_capacity:
        print(
            "WARNING: there might be more trains than stations can hold")  # TODO: take this into account when assigning capacities to stations?

    # keep track of some randomly generated values; TODO: actually use these
    # to fulfill todos in sanity check instead of just giving a warning
    # TOTAL_STATION_CAPACITY = 0
    # TOTAL_TRAIN_CAPACITY = 0

    # use args to generate the world
    graph = random_connected_graph(['S' + str(i + 1)
                                    for i in range(num_stations)], density)
    stations = {}
    lines = {}
    trains = {}
    for node in graph.nodes:
        station_capacity = random.randint(
            min_station_capacity, max_station_capacity)
        # TOTAL_STATION_CAPACITY += station_capacity
        stations[node] = Station(node, station_capacity, [], [])

    for edge in graph.edges:
        length = random.randint(
            min_line_length, max_line_length)
        graph.edges[edge]['weight'] = length
        name = graph.edges[edge]['name']
        capacity = random.randint(min_line_capacity, max_line_capacity)
        lines[name] = Line(name, length, stations[edge[0]],
                           stations[edge[1]], capacity, [])

    for i in range(num_trains):
        train_capacity = random.randint(
            min_train_capacity, max_train_capacity)
        # TOTAL_TRAIN_CAPACITY += train_capacity
        name = 'T' + str(i + 1)
        start = random.choice([*graph.nodes, '*'])
        speed = round(
            random.uniform(
                min_train_speed,
                max_train_speed),
            5)
        if start != '*':
            trains[name] = Train(
                name,
                stations[start],
                TrainPositionType.STATION,
                speed,
                train_capacity,
                [])
            stations[start].trains.append(trains[name])
        else:
            trains[name] = Train(
                name,
                None,
                TrainPositionType.NOT_STARTED,
                speed,
                train_capacity,
                [])
    passengers = {}
    for i in range(num_passengers):
        name = 'P' + str(i + 1)
        start = stations[random.choice(tuple(graph.nodes))]
        destination = stations[random.choice(tuple(graph.nodes))]
        size = random.randint(
            min_group_size, max_group_size)
        time = random.randint(min_time, max_time)
        passengers[name] = PassengerGroup(
            name,
            start,
            PassengerGroupPositionType.STATION,
            size,
            destination,
            time)

    new_game_state = GameState(trains, passengers, stations, lines)
    return new_game_state, graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?', default=None)
    parser.add_argument('--density', default=0, type=float,
                        help='Network density, approx. |E|/|V|^2. 0 is just connected, 1 is fully connected.')

    # stations
    parser.add_argument('--num-stations', type=int)
    parser.add_argument('--min-station-capacity', type=int)
    parser.add_argument('--max-station-capacity', type=int)

    # lines
    parser.add_argument('--min-line-capacity', type=int)
    parser.add_argument('--max-line-capacity', type=int)
    parser.add_argument('--min-line-length', type=int)
    parser.add_argument('--max-line-length', type=int)

    # trains
    parser.add_argument('--num-trains', type=int)
    parser.add_argument('--min-train-speed', type=int)
    parser.add_argument('--max-train-speed', type=int)
    parser.add_argument('--min-train-capacity', type=int)
    parser.add_argument('--max-train-capacity', type=int)

    # passengers
    parser.add_argument('--num-passengers', type=int)
    parser.add_argument('--min-group-size', type=int)
    parser.add_argument('--max-group-size', type=int)
    parser.add_argument('--min-time', type=int)
    parser.add_argument('--max-time', type=int)
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()

    args_dict = vars(args)

    game_state, network_graph = generate_game_state(
        **{key: args_dict[key] for key in args_dict if args_dict[key] is not None})

    if args.output is None:
        print(game_state.serialize())
    else:
        with open(args.output, 'w') as f:
            f.write(game_state.serialize())

    if args.draw:
        nx.draw(network_graph, with_labels=True)
        plt.show()
