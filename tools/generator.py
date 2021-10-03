import math
import random
import argparse

import networkx as nx
import matplotlib.pyplot as plt


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


def print_example(graph: nx.Graph, trains: dict, passengers: dict, path=None):  # pylint: disable=redefined-outer-name
    if path is not None:
        file = open(path, 'w', encoding='utf-8')
    else:
        file = None
    print(f"#total_station_capacity: {TOTAL_STATION_CAPACITY}", file=file)
    print("[Stations]", file=file)
    for current_node in graph.nodes:
        print(current_node, graph.nodes[current_node]['capacity'], file=file)
    print(file=file)
    print("[Lines]", file=file)
    for current_edge in graph.edges:
        print(graph.edges[current_edge]['name'], current_edge[0], current_edge[1], graph.edges[current_edge]
              ['length'], graph.edges[current_edge]['capacity'], file=file)
    print(file=file)
    print(f"#total_train_capacity: {TOTAL_TRAIN_CAPACITY}", file=file)
    print("[Trains]", file=file)
    for train in trains:
        print(train, *trains[train], file=file)
    print(file=file)
    print("[Passengers]", file=file)
    for passenger in passengers:
        print(passenger, *passengers[passenger], file=file)

    if path is not None:
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?', default=None)
    parser.add_argument('--density', default=0, type=float,
                        help='Network density, approx. |E|/|V|^2. 0 is just connected, 1 is fully connected.')

    # stations
    parser.add_argument('--num-stations', type=int)
    parser.add_argument('--min-station-capacity', default=1, type=int)
    parser.add_argument('--max-station-capacity', default=2, type=int)

    # lines
    parser.add_argument('--min-line-capacity', default=1, type=int)
    parser.add_argument('--max-line-capacity', default=2, type=int)
    parser.add_argument('--min-line-length', default=1, type=int)
    parser.add_argument('--max-line-length', default=5, type=int)

    # trains
    parser.add_argument('--num-trains', type=int)
    parser.add_argument('--min-train-speed', default=1, type=int)
    parser.add_argument('--max-train-speed', default=5, type=int)
    parser.add_argument('--min-train-capacity', default=5, type=int)
    parser.add_argument('--max-train-capacity', default=10, type=int)

    # passengers
    parser.add_argument('--num-passengers', default=5, type=int)
    parser.add_argument('--min-group-size', default=1, type=int)
    parser.add_argument('--max-group-size', default=5, type=int)
    parser.add_argument('--min-time', default=1, type=int)
    parser.add_argument('--max-time', default=10, type=int)
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()

    # trying to make world more realistic by "dynamically" adjusting unfilled parameters with respect to the given constraints
    # step 1: set num_stations if needed
    if args.num_stations is None:
        if args.num_trains is None:
            args.num_trains = random.randint(1, 10)
        # set num_stations and num_trains roughly to the same value (+-20%)
        args.num_stations = random.randint(
            max(5, math.floor(args.num_trains * 0.8)), math.ceil(args.num_trains * 1.2))
    # step 2: set num_trains if needed
    if args.num_trains is None:
        # set num_stations and num_trains roughly to the same value (+-20%)
        args.num_trains = random.randint(
            max(1, math.floor(args.num_stations * 0.8)), math.ceil(args.num_stations * 1.2))

    # some sanity checks
    if args.min_train_capacity < args.max_group_size:
        print(
            "WARNING: passenger groups might be too large to be transported by any train")  # TODO: take this into account when assigning train capacities / group sizes?
    if args.num_trains > args.num_stations * args.min_station_capacity:
        print(
            "WARNING: there might be more trains than stations can hold")  # TODO: take this into account when assigning capacities to stations?

    # keep track of some randomly generated values; TODO: actually use these
    # to fulfill todos in sanity check instead of just giving a warning
    TOTAL_STATION_CAPACITY = 0
    TOTAL_TRAIN_CAPACITY = 0

    # use args to generate the world
    G = random_connected_graph(['S' + str(i + 1)
                               for i in range(args.num_stations)], args.density)
    for node in G.nodes:
        station_capacity = random.randint(
            args.min_station_capacity, args.max_station_capacity)
        TOTAL_STATION_CAPACITY += station_capacity
        G.nodes[node]['capacity'] = station_capacity
    for edge in G.edges:
        G.edges[edge]['capacity'] = random.randint(
            args.min_line_capacity, args.max_line_capacity)
        G.edges[edge]['length'] = random.randint(
            args.min_line_length, args.max_line_length)

    trains = {}
    for i in range(args.num_trains):
        train_capacity = random.randint(
            args.min_train_capacity, args.max_train_capacity)
        TOTAL_TRAIN_CAPACITY += train_capacity
        trains['T' + str(i + 1)] = (
            random.choice([*G.nodes, '*']),
            round(
                random.uniform(
                    args.min_train_speed,
                    args.max_train_speed),
                5),
            train_capacity
        )
    passengers = {}
    for i in range(args.num_passengers):
        passengers['P' + str(i + 1)] = (random.choice(tuple(G.nodes)), random.choice(tuple(G.nodes)),
                                        random.randint(
                                            args.min_group_size, args.max_group_size),
                                        random.randint(args.min_time, args.max_time))

    print_example(G, trains, passengers, args.output)

    if args.draw:
        nx.draw(G, with_labels=True)
        plt.show()
