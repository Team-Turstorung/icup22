from abc import ABC, abstractmethod

import networkx as nx

from abfahrt.types import NetworkState, Schedule


class Solution(ABC):
    def __init__(self, network_state: NetworkState, network_graph: nx.Graph):
        self.network_state = network_state
        self.network_graph = network_graph

    @abstractmethod
    def schedule(self) -> Schedule:
        pass
