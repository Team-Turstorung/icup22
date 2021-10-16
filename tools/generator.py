import math
import random
import argparse
from typing import Tuple

import networkx as nx

from abfahrt.types import NetworkState, Station, Line, Train, TrainPositionType, PassengerGroup, \
    PassengerGroupPositionType


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


def generate_stations_and_lines(**kwargs) -> Tuple[NetworkState, nx.Graph]:
    num_stations = kwargs.get('num_stations', random.randint(5, 10))
    density = kwargs.get('density', 0)
    min_station_capacity = kwargs.get('min_station_capacity', 1)
    max_station_capacity = kwargs.get('max_station_capacity', 5)
    min_line_length = kwargs.get('min_line_length', 1)
    max_line_length = kwargs.get('max_line_length', 10)
    min_line_capacity = kwargs.get('min_line_capacity', 1)
    max_line_capacity = kwargs.get('max_line_capacity', 5)

    graph = random_connected_graph(['S' + str(i + 1) for i in range(num_stations)], density)

    stations = {}
    lines = {}
    for node in graph.nodes:
        station_capacity = random.randint(
            min_station_capacity, max_station_capacity)
        stations[node] = Station(
            name=node,
            capacity=station_capacity,
            trains=[],
            passenger_groups=[])

    for edge in graph.edges:

        length = round(
            random.uniform(
                min_line_length,
                max_line_length),
            5)
        graph.edges[edge]['weight'] = length
        name = graph.edges[edge]['name']
        capacity = random.randint(min_line_capacity, max_line_capacity)
        lines[name] = Line(name=name, length=length, start=edge[0],
                           end=edge[1], capacity=capacity, trains=[])
    return NetworkState(stations=stations, lines=lines, passenger_groups={}, trains={}), graph


def generate_trains(state: NetworkState, graph: nx.Graph, **kwargs):
    num_trains = kwargs.get('num_trains', random.randint(1, math.ceil(len(state.stations) * 0.8)))
    min_train_speed = kwargs.get('min_train_speed', 1)
    max_train_speed = kwargs.get('max_train_speed', 10)
    min_train_capacity = kwargs.get('min_train_capacity', 1)
    max_train_capacity = kwargs.get('max_train_capacity', 60)
    trains = {}
    for i in range(num_trains):
        train_capacity = random.randint(
            min_train_capacity, max_train_capacity)
        name = 'T' + str(i + 1)
        start = random.choice([*graph.nodes, '*'])
        speed = round(
            random.uniform(
                min_train_speed,
                max_train_speed),
            5)
        if start != '*':
            trains[name] = Train(
                name=name,
                position=start,
                position_type=TrainPositionType.STATION,
                speed=speed,
                capacity=train_capacity,
                passenger_groups=[],
                next_station=None)
            state.stations[start].trains.append(name)
        else:
            trains[name] = Train(
                name=name,
                position=None,
                position_type=TrainPositionType.NOT_STARTED,
                speed=speed,
                capacity=train_capacity,
                passenger_groups=[],
                next_station=None)
    state.trains = trains


def generate_passenger_groups(state: NetworkState, graph: nx.Graph, **kwargs):
    num_passengers = kwargs.get(
        'num_passengers', random.randint(
            1, math.floor(
                len(state.stations) * 0.8)))
    min_group_size = kwargs.get('min_group_size', 1)
    max_group_size = kwargs.get('min_group_size', 15)
    min_time = kwargs.get('min_time', 1)
    max_time = kwargs.get('min_time', 10)
    passengers = {}
    for i in range(num_passengers):
        name = 'P' + str(i + 1)
        start = random.choice(tuple(graph.nodes))
        destination = random.choice(tuple(graph.nodes))
        size = random.randint(
            min_group_size, max_group_size)
        time = random.randint(min_time, max_time)
        passengers[name] = PassengerGroup(
            name=name,
            position=start,
            position_type=PassengerGroupPositionType.STATION,
            group_size=size,
            destination=destination,
            time_remaining=time)
    state.passenger_groups = passengers


def generate_game_state(**kwargs) -> Tuple[NetworkState, nx.Graph]:
    for _ in range(5):
        state, graph = generate_stations_and_lines(**kwargs)
        generate_trains(state, graph, **kwargs)
        generate_passenger_groups(state, graph, **kwargs)

        if not state.is_valid():
            print("state is invalid, regenerating...")
            continue
        max_group_size = max([group.group_size for group in state.passenger_groups.values()])
        max_train_capacity = max([train.capacity for train in state.trains.values()])
        if max_group_size > max_train_capacity:
            print("there are groups that cannot be transported, regenerating...")
            continue
        return state, graph
    raise Exception("could not generate a reasonable network after 5 attempts. please check your inṕut parameters!")


def create_generator_parser(parser: argparse.ArgumentParser):
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
    parser.add_argument('--min-line-length', type=float)
    parser.add_argument('--max-line-length', type=float)

    # trains
    parser.add_argument('--num-trains', type=int)
    parser.add_argument('--min-train-speed', type=float)
    parser.add_argument('--max-train-speed', type=float)
    parser.add_argument('--min-train-capacity', type=int)
    parser.add_argument('--max-train-capacity', type=int)

    # passengers
    parser.add_argument('--num-passengers', type=int)
    parser.add_argument('--min-group-size', type=int)
    parser.add_argument('--max-group-size', type=int)
    parser.add_argument('--min-time', type=int)
    parser.add_argument('--max-time', type=int)


def execute(args: argparse.Namespace):
    args_dict = vars(args)
    game_state, _ = generate_game_state(
        **{key: args_dict[key] for key in args_dict if args_dict[key] is not None})
    if args.output is None:
        print(game_state.serialize())
    else:
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(game_state.serialize())
