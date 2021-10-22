from dataclasses import field, dataclass, asdict
import logging
from typing import Dict, List, Set, Tuple
from itertools import combinations

import networkx as nx
from networkx.algorithms import all_pairs_dijkstra

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction, Train, PassengerGroup, Station, Line, \
    PassengerGroupPositionType


@dataclass
class MultiTrain(Train):
    on_tour: bool = False
    path: List[str] = field(default_factory=list)
    reserved_capacity: int = 0
    assigned_passenger_groups: Set[str] = field(default_factory=set)
    is_blocked: bool = False


@dataclass
class MultiLine(Line):
    reserved_capacity: int = 0


@dataclass
class MultiStation(Station):
    locks: Set[str] = field(default_factory=set)


@dataclass
class MultiPassengerGroup(PassengerGroup):
    is_assigned: bool = False
    priority: int = 0


@dataclass
class MultiNetworkState(NetworkState):
    trains: Dict[str, MultiTrain] = field(default_factory=dict)
    passenger_groups: Dict[str, MultiPassengerGroup] = field(default_factory=dict)
    stations: Dict[str, MultiStation] = field(default_factory=dict)
    lines: Dict[str, MultiLine] = field(default_factory=dict)

    def waiting_passengers(self) -> Dict[str, MultiPassengerGroup]:
        return {name: group for name, group in self.passenger_groups.items() if not group.is_destination_reached() and group.position_type == PassengerGroupPositionType.STATION}


def multify_network(network_state: NetworkState) -> MultiNetworkState:
    new_state = MultiNetworkState()
    for station_id, station in network_state.stations.items():
        new_state.stations[station_id] = MultiStation(**asdict(station))
    for line_id, line in network_state.lines.items():
        new_state.lines[line_id] = MultiLine(**asdict(line))
    for train_id, train in network_state.trains.items():
        new_state.trains[train_id] = MultiTrain(**asdict(train))
    for passenger_group_id, passenger_group in network_state.passenger_groups.items():
        new_state.passenger_groups[passenger_group_id] = MultiPassengerGroup(**asdict(passenger_group))
    return new_state


class SimplesSolverMultipleTrains(Solution):
    def get_all_shortest_paths(self, network_graph: nx.Graph) -> Dict[str, Tuple]:
        shortest_paths = all_pairs_dijkstra(network_graph)
        all_shortest_paths = {}
        for path in shortest_paths:
            all_shortest_paths[path[0]] = path[1]
        return all_shortest_paths

    def place_wildcard_trains(self, network_state: MultiNetworkState) -> RoundAction:

        new_round_action = RoundAction()

        wildcard_trains = filter(lambda train: train.position_type == TrainPositionType.NOT_STARTED,
                                 network_state.trains.values())
        sorted_wildcard_trains = sorted(wildcard_trains, key=lambda train: train.speed, reverse=True)
        passenger_list = sorted(network_state.passenger_groups.values(), reverse=True, key=lambda passenger: passenger.priority)

        station_space_left = dict()
        for station_id, station in network_state.stations.items():
            station_space_left[station_id] = station.capacity - len(station.trains)

        for current_passenger_group in passenger_list:
            station = network_state.stations[current_passenger_group.position]
            if station_space_left[station.name] <= 0:
                continue

            max_speed = 0
            for current_train_name in station.trains:
                current_train = network_state.trains[current_train_name]
                if current_train.capacity >= current_passenger_group.group_size:
                    max_speed = max(max_speed, current_train.speed)

            for current_train in sorted_wildcard_trains:
                if current_train.capacity >= current_passenger_group.group_size:
                    # Check if there already is a faster train
                    if max_speed >= current_train.speed:
                        break
                    # Place train here
                    new_round_action.train_starts[current_train.name] = station.name
                    sorted_wildcard_trains.remove(current_train)
                    station_space_left[station.name] -= 1
                    break

            if len(sorted_wildcard_trains) == 0:
                break
        if len(sorted_wildcard_trains) != 0:
            for current_train in network_state.trains.values():
                if current_train.position_type == TrainPositionType.NOT_STARTED:
                    emptiest_station = max(
                        station_space_left.items(),
                        key=lambda item: item[1])[0]
                    station_space_left[emptiest_station] -= 1
                    new_round_action.train_starts[current_train.name] = emptiest_station
        return new_round_action

    def compute_priorities(self, network_state: MultiNetworkState, all_shortest_paths: dict[str, Tuple]):
        for passenger_group in network_state.passenger_groups.values():
            passenger_group.priority = all_shortest_paths[passenger_group.position][0][
                                                   passenger_group.destination] / (
                                                   passenger_group.time_remaining + 1) * passenger_group.group_size

    def navigate_train(self, network_state: MultiNetworkState, network_graph: nx.Graph, all_shortest_paths: Dict[str, Tuple], train: MultiTrain, round_action: RoundAction):
        if train.position_type == TrainPositionType.LINE or not train.on_tour:
            return
        # Check if there is a passenger detrained
        pause_train = False
        for passenger_group_name in train.passenger_groups:
            passenger_group = network_state.passenger_groups[passenger_group_name]
            if passenger_group.destination == train.position or all_shortest_paths[train.position][1][passenger_group.destination][0:2] != train.path[0:2]:
                round_action.passenger_detrains.append(passenger_group.name)
                pause_train = True

        network_state.stations[train.position].locks.discard(train.name)
        # We reached our current destination
        if len(train.path) == 1:
            # We have to pause cause we need a new destination first TODO: Look for new destination if we don't board or detrain
            pause_train = True
            # If there is reserved capacity we want to pickup a new passenger
            if train.reserved_capacity != 0:
                passenger_groups = [network_state.passenger_groups[passenger_group_id] for passenger_group_id in network_state.stations[train.position].passenger_groups]
                if len(passenger_groups) > 0:
                    for passenger_group in sorted(passenger_groups, key=lambda group: group.priority):
                        if not passenger_group.is_destination_reached() and passenger_group.name in train.assigned_passenger_groups:
                            train.assigned_passenger_groups.remove(passenger_group.name)
                            passenger_group.is_assigned = False
                            round_action.passenger_boards[passenger_group.name] = train.name
                            break
            train.on_tour = False

        current_station = network_state.stations[train.position]
        # Check if we can take a new passenger with us
        if len(current_station.passenger_groups) != 0:
            free_capacity = self.calculate_free_capacity(network_state, train)
            for new_passenger_group_name in current_station.passenger_groups:
                new_passenger_group = network_state.passenger_groups[new_passenger_group_name]
                # Check if there is no other train that wants the passenger, there is enough space and it is the right direction
                if not new_passenger_group.is_assigned and free_capacity >= new_passenger_group.group_size and not new_passenger_group.is_destination_reached():
                    if all_shortest_paths[train.position][1][new_passenger_group.destination][0:2] == train.path[0:2]:
                        train.assigned_passenger_groups.add(new_passenger_group.name)
                        new_passenger_group.is_assigned = True
                        # Because we board a new passenger we can't go on this round
                        pause_train = True
                        round_action.passenger_boards[new_passenger_group.name] = train.name
                        free_capacity -= new_passenger_group.group_size
        if pause_train:
            return

        # go to next station in path
        next_line_id = network_graph.edges[train.position, train.path[1]]['name']
        next_line = network_state.lines[next_line_id]
        if (next_line.reserved_capacity + len(next_line.trains) - next_line.capacity) < 0 or next_line.length <= train.speed:
            next_station = network_state.stations[train.path[1]]
            if len(next_station.locks) + len(next_station.trains) - next_station.capacity < 0:
                next_line.reserved_capacity += 1
                next_station.locks.add(train.name)
                round_action.train_departs[train.name] = next_line_id
                train.path = train.path[1:]
            else:
                train.is_blocked = True

    def calculate_free_capacity(self, network_state: MultiNetworkState, train: MultiTrain) -> int:
        # Get the capacity that is left on a train, to decide if we want new passengers.
        occupied_space = 0
        for passenger_group_name in train.passenger_groups:
            passenger_group = network_state.passenger_groups[passenger_group_name]
            occupied_space += passenger_group.group_size
        return train.capacity - occupied_space - train.reserved_capacity

    def plan_train(self, network_state: MultiNetworkState, network_graph: nx.Graph, all_shortest_paths: dict[str, Tuple], train: MultiTrain):
        if len(train.passenger_groups) == 0:
            # go to suitable passenger group
            passengers_sorted_by_priority = sorted(network_state.waiting_passengers().values(),
                                                   key=lambda passenger_group: passenger_group.priority / (
                                                       all_shortest_paths[train.position][0][passenger_group.position] + 1),
                                                   reverse=True)

            for passenger_group in passengers_sorted_by_priority:
                if not passenger_group.is_assigned and passenger_group.group_size <= train.capacity:
                    # Select matching passenger group for train
                    train.assigned_passenger_groups.add(passenger_group.name)
                    passenger_group.is_assigned = True
                    train.reserved_capacity = passenger_group.group_size
                    return all_shortest_paths[train.position][1][passenger_group.position]

            if network_state.stations[train.position].is_full():
                for neighbor in network_graph.neighbors(train.position):
                    if not network_state.stations[neighbor].is_full():
                        return [train.position, neighbor]

        else:
            # go to destination of passenger group with highest priority
            passenger_group = network_state.passenger_groups[train.passenger_groups[0]]
            for passenger_group_name in train.passenger_groups[1:]:
                current_passenger_group = network_state.passenger_groups[passenger_group_name]
                if current_passenger_group.priority > passenger_group.priority:
                    passenger_group = current_passenger_group
            train.reserved_capacity = 0
            return all_shortest_paths[train.position][1][passenger_group.destination]

    def schedule(self, network_state: NetworkState, network_graph: nx.Graph) -> Schedule:

        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)

        network_state: MultiNetworkState = multify_network(network_state)

        all_shortest_paths = self.get_all_shortest_paths(network_graph)
        self.compute_priorities(network_state, all_shortest_paths)

        # Create round action for zero Round
        round_action = self.place_wildcard_trains(network_state)

        actions = dict()
        round_id = 0
        actions[round_id] = round_action
        network_state.apply(round_action)

        # Game loop, till there are no more passengers to transport
        while True:
            print(f"Processing round {round_id}")
            for line in network_state.lines.values():
                line.reserved_capacity = 0
            for train in network_state.trains.values():
                train.is_blocked = False
            round_action = RoundAction()
            round_id += 1
            for train in sorted(network_state.trains.values(), key=lambda train: train.speed, reverse=True):
                if train.on_tour:
                    continue
                path = self.plan_train(network_state, network_graph, all_shortest_paths, train)
                if path is not None:
                    train.on_tour = True
                    train.path = path

            for train in network_state.trains.values():
                self.navigate_train(network_state, network_graph, all_shortest_paths, train, round_action)

            processed = set()
            for pair in combinations([train.name for train in network_state.trains.values() if train.is_blocked], 2):
                train1 = network_state.trains[pair[0]]
                train2 = network_state.trains[pair[1]]
                if train1.position == train2.position or train1.name in processed or train2.name in processed:
                    continue
                set1 = set(train1.path[0:2])
                set2 = set(train2.path[0:2])
                common_path = set1.intersection(set2)
                if len(common_path) == 2:
                    next_line_id = network_graph.edges[train1.position, train1.path[1]]['name']
                    next_line = network_state.lines[next_line_id]
                    if ((next_line.reserved_capacity + len(next_line.trains) - next_line.capacity) < 0 and (next_line.length <= train1.speed or next_line.length <= train2.speed)) or ((next_line.reserved_capacity + len(next_line.trains) - next_line.capacity) < -1) or (next_line.length <= train1.speed and next_line.length <= train2.speed):
                        network_state.stations[train1.position].locks.add(train2.name)
                        network_state.stations[train2.position].locks.add(train1.name)
                        # TODO: differentiate between possible cases
                        next_line.reserved_capacity += 2
                        round_action.train_departs[train1.name] = next_line_id
                        round_action.train_departs[train2.name] = next_line_id
                        train1.path = train1.path[1:]
                        train2.path = train2.path[1:]
                        processed.add(train1.name)
                        processed.add(train2.name)

            actions[round_id] = round_action
            network_state.apply(round_action)
            if not network_state.is_valid():
                raise Exception(f"invalid state at round {round_id}")
            if network_state.is_finished():
                break
        schedule = Schedule.from_dict(actions)
        return schedule
