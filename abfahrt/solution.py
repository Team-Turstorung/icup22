from abc import ABC, abstractmethod

import networkx as nx

from abfahrt.types import NetworkState, Schedule


class Solution(ABC):
    @abstractmethod
    def schedule(self, network_state: NetworkState,
                 network_graph: nx.Graph) -> Schedule:
        pass
