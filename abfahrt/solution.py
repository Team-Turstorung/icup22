from abc import ABC, abstractmethod

import networkx as nx

from abfahrt.types import NetworkState, Schedule


class Solution(ABC):
    @abstractmethod
    def schedule(self, game_state: NetworkState, graph: nx.Graph) -> Schedule:
        pass
