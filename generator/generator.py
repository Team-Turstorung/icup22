import random

import networkx as nx
import argparse
import matplotlib.pyplot as plt


# adapted from https://stackoverflow.com/a/14618505
def random_connected_graph(nodes, density=0):
    if density >= 1:
        return nx.complete_graph(nodes)
    # Create two partitions, S and T. Initially store all nodes in S.
    S, T = set(nodes), set()
    edge_counter = 1

    # Pick a random node, and mark it as visited and the current node.
    current_node = random.choice(nodes)
    S.remove(current_node)
    T.add(current_node)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    # Create a random connected graph.
    while S:
        # Randomly pick the next node from the neighbors of the current node.
        # As we are generating a connected graph, we assume a complete graph.
        neighbor_node = random.choice(nodes)
        # If the new node hasn't been visited, add the edge from current to new.
        if neighbor_node not in T:
            graph.add_edge(current_node, neighbor_node, name='L' + str(edge_counter))
            S.remove(neighbor_node)
            T.add(neighbor_node)
            edge_counter += 1
        # Set the new node as the current node.
        current_node = neighbor_node

    missing_edges = (graph.number_of_nodes() * (graph.number_of_nodes() - 1) * density) // 2 - graph.number_of_edges()
    while missing_edges > 0:
        n1, n2 = random.choice(nodes), random.choice(nodes)
        if not graph.has_edge(n1, n2) and not graph.has_edge(n2, n1) and n1 != n2:
            graph.add_edge(n1, n2, name='L' + str(edge_counter))
            edge_counter += 1
            missing_edges -= 1

    return graph


def print_example(G: nx.Graph, trains, passengers, path=None):
    if path is not None:
        f = open(path, 'w')
    else:
        f = None
    print("[Stations]", file=f)
    for node in G.nodes:
        print(node, G.nodes[node]['capacity'], file=f)
    print(file=f)
    print("[Lines]", file=f)
    for edge in G.edges:
        print(G.edges[edge]['name'], edge[0], edge[1], G.edges[edge]['length'], G.edges[edge]['capacity'], file=f)
    print(file=f)
    print("[Trains]", file=f)
    for train in trains:
        print(train, *trains[train], file=f)
    print(file=f)

    print("[Passengers]", file=f)
    for passenger in passengers:
        print(passenger, *passengers[passenger], file=f)

    if path is not None:
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?', default=None)
    parser.add_argument('--stations', default=5, type=int)
    parser.add_argument('--density', default=0, type=float,
                        help='Network density, approx. |E|/|V|^2. 0 is just connected, 1 is fully connected.')
    parser.add_argument('--min-station-capacity', default=1, type=int)
    parser.add_argument('--max-station-capacity', default=2, type=int)
    parser.add_argument('--min-line-capacity', default=1, type=int)
    parser.add_argument('--max-line-capacity', default=2, type=int)
    parser.add_argument('--min-line-length', default=1, type=int)
    parser.add_argument('--max-line-length', default=5, type=int)
    parser.add_argument('--num-trains', default=3, type=int)
    parser.add_argument('--min-train-speed', default=1, type=int)
    parser.add_argument('--max-train-speed', default=5, type=int)
    parser.add_argument('--min-train-capacity', default=1, type=int)
    parser.add_argument('--max-train-capacity', default=2, type=int)
    parser.add_argument('--num-passengers', default=5, type=int)
    parser.add_argument('--min-group-size', default=1, type=int)
    parser.add_argument('--max-group-size', default=5, type=int)
    parser.add_argument('--min-time', default=1, type=int)
    parser.add_argument('--max-time', default=10, type=int)
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()

    G = random_connected_graph(['S' + str(i + 1) for i in range(args.stations)], args.density)
    for node in G.nodes:
        G.nodes[node]['capacity'] = random.randint(args.min_station_capacity, args.max_station_capacity)
    for edge in G.edges:
        G.edges[edge]['capacity'] = random.randint(args.min_line_capacity, args.max_line_capacity)
        G.edges[edge]['length'] = random.randint(args.min_line_length, args.max_line_length)

    trains = {}
    for i in range(args.num_trains):
        trains['T' + str(i + 1)] = (
            random.choice([*G.nodes, '*']), random.randint(args.min_train_speed, args.max_train_speed),
            random.randint(args.min_train_capacity, args.max_train_capacity)
        )
    passengers = {}
    for i in range(args.num_passengers):
        passengers['P' + str(i + 1)] = (random.choice(tuple(G.nodes)), random.choice(tuple(G.nodes)),
                                        random.randint(args.min_group_size, args.max_group_size),
                                        random.randint(args.min_time, args.max_time))

    print_example(G, trains, passengers, args.output)

    if args.draw:
        nx.draw(G, with_labels=True)
        plt.show()
