import random
import networkx as nx
import argparse
import matplotlib.pyplot as plt

random.seed(1)


def random_connected_graph(nodes, density=0):
    if density >= 1:
        return nx.complete_graph(nodes)
    # Create two partitions, S and T. Initially store all nodes in S.
    S, T = set(nodes), set()

    # Pick a random node, and mark it as visited and the current node.
    current_node = random.choice(nodes)
    S.remove(current_node)
    T.add(current_node)

    graph = nx.Graph(nodes=nodes)

    # Create a random connected graph.
    while S:
        # Randomly pick the next node from the neighbors of the current node.
        # As we are generating a connected graph, we assume a complete graph.
        neighbor_node = random.choice(nodes)
        # If the new node hasn't been visited, add the edge from current to new.
        if neighbor_node not in T:
            graph.add_edge(current_node, neighbor_node)
            S.remove(neighbor_node)
            T.add(neighbor_node)
        # Set the new node as the current node.
        current_node = neighbor_node

    missing_edges = (graph.number_of_nodes()*(graph.number_of_nodes()-1) * density)//2 - graph.number_of_edges()
    print(graph, missing_edges)
    while missing_edges > 0:
        n1, n2 = random.choice(nodes), random.choice(nodes)
        if not graph.has_edge(n1, n2) and not graph.has_edge(n2, n1) and n1 != n2:
            graph.add_edge(n1, n2)
            missing_edges -= 1

    return graph


parser = argparse.ArgumentParser()
parser.add_argument('--stations', default=5, type=int)
parser.add_argument('--density', default=0, type=float)
args = parser.parse_args()

G = random_connected_graph(['S'+str(i+1) for i in range(args.stations)], args.density)
nx.draw(G, with_labels=True)
plt.show()
