from copy import deepcopy
from dataclasses import field, dataclass, asdict
import enum
import logging
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations

import networkx as nx
from networkx.algorithms import all_pairs_dijkstra

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction, Train, PassengerGroup, Station, Line, \
    PassengerGroupPositionType


class TrainState(enum.Enum):
    BOARDING = 0
    BLOCKED_STATION = 1
    DEPARTING = 2
    UNUSED = 3
    BLOCKED_LINE = 4
    ARRIVING = 5
    WAITING_FOR_SWAP = 6

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass(unsafe_hash=True)
class MultiTrain(Train):
    path: List[str] = field(default_factory=list, compare=False)
    assigned_passenger_group: Optional['MultiPassengerGroup'] = None
    station_state: TrainState = TrainState.UNUSED

    @property
    def reserved_capacity(self):
        return 0 if self.assigned_passenger_group is None or self.assigned_passenger_group.position == self.name else self.assigned_passenger_group.group_size


@dataclass
class MultiLine(Line):
    reserved_capacity: int = 0


@dataclass
class MultiStation(Station):
    locks: Set[str] = field(default_factory=set)


@dataclass(unsafe_hash=True)
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
        return {name: group for name, group in self.passenger_groups.items() if
                not group.is_destination_reached() and group.position_type == PassengerGroupPositionType.STATION}


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


class SimpleSolverMultipleTrains(Solution):

    def __init__(self, network_state: NetworkState, network_graph: nx.Graph):
        super().__init__(network_state, network_graph)
        self.all_shortest_paths = self.get_all_shortest_paths()
        self.network_state = multify_network(self.network_state)

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def get_all_shortest_paths(self) -> Dict[str, Tuple]:
        shortest_paths = all_pairs_dijkstra(self.network_graph)
        all_shortest_paths = {}
        for path in shortest_paths:
            all_shortest_paths[path[0]] = path[1]
        return all_shortest_paths

    def place_wildcard_trains(self) -> RoundAction:
        new_round_action = RoundAction()

        wildcard_trains = filter(lambda train: train.position_type == TrainPositionType.NOT_STARTED,
                                 self.network_state.trains.values())
        sorted_wildcard_trains = sorted(wildcard_trains, key=lambda train: train.speed, reverse=True)
        passenger_list = sorted(self.network_state.passenger_groups.values(), reverse=True,
                                key=lambda passenger: passenger.priority)

        station_space_left = dict()
        for station_id, station in self.network_state.stations.items():
            station_space_left[station_id] = station.capacity - len(station.trains)

        for current_passenger_group in passenger_list:
            station = self.network_state.stations[current_passenger_group.position]
            if station_space_left[station.name] <= 0:
                continue

            max_speed = 0
            for current_train_name in station.trains:
                current_train = self.network_state.trains[current_train_name]
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
            for current_train in self.network_state.trains.values():
                if current_train.position_type == TrainPositionType.NOT_STARTED:
                    emptiest_station = max(
                        station_space_left.items(),
                        key=lambda item: item[1])[0]
                    if station_space_left[emptiest_station] > 1:
                        station_space_left[emptiest_station] -= 1
                        new_round_action.train_starts[current_train.name] = emptiest_station
                    else:
                        break
        return new_round_action

    def compute_priorities(self):
        for passenger_group in self.network_state.passenger_groups.values():
            passenger_group.priority = self.all_shortest_paths[passenger_group.position][0][
                                           passenger_group.destination] / (
                                           passenger_group.time_remaining + 1) * passenger_group.group_size

    def calculate_free_capacity(self, train: MultiTrain) -> int:
        # Get the capacity that is left on a train, to decide if we want new passengers.
        occupied_space = 0
        for passenger_group_name in train.passenger_groups:
            passenger_group = self.network_state.passenger_groups[passenger_group_name]
            occupied_space += passenger_group.group_size
        return train.capacity - occupied_space - train.reserved_capacity

    def update_route_for_boarded_passenger(self, train: MultiTrain):
        train.path = self.all_shortest_paths[train.position][1][train.assigned_passenger_group.destination]

    def update_route_to_assigned_passenger(self, train: MultiTrain):
        train.path = self.all_shortest_paths[train.position][1][train.assigned_passenger_group.position]

    def update_route_prevent_full_stations(self, train: MultiTrain, round_action: RoundAction):
        assert train.reserved_capacity == 0
        assert len([passenger_group for passenger_group in train.passenger_groups if
                    passenger_group not in round_action.passenger_detrains and self.network_state.passenger_groups[passenger_group].destination != train.position]) == 0
        current_position = self.network_state.stations[train.position]
        if len(current_position.locks) + len(current_position.trains) >= current_position.capacity:
            for neighbor in self.network_graph.neighbors(train.position):
                neighbor_station = self.network_state.stations[neighbor]
                if len(neighbor_station.locks) + len(neighbor_station.trains) < neighbor_station.capacity:
                    train.path = [train.position, neighbor]
                    return
            for neighbor in self.network_graph.neighbors(train.position):
                neighbor_station = self.network_state.stations[neighbor]
                for swap_train in [self.network_state.trains[train_name] for train_name in neighbor_station.trains]:
                    if len(swap_train.path) > 1 and swap_train.path[1] == train.position and swap_train.assigned_passenger_group is not None:
                        train.path = [train.position, neighbor]
                        return

    def reserve_next_passenger(self, train: MultiTrain, round_action: RoundAction) -> Optional[MultiPassengerGroup]:
        assert train.assigned_passenger_group is None
        if len([passenger_group for passenger_group in train.passenger_groups if
                passenger_group not in round_action.passenger_detrains and self.network_state.passenger_groups[passenger_group].destination != train.position]) != 0:
            train.assigned_passenger_group = max(
                [self.network_state.passenger_groups[passenger_group] for passenger_group in train.passenger_groups if passenger_group not in round_action.passenger_detrains and self.network_state.passenger_groups[passenger_group].destination != train.position],
                key=lambda passenger_group: passenger_group.priority)
            train.assigned_passenger_group.is_assigned = False
            return train.assigned_passenger_group

        passengers_sorted_by_priority = sorted(self.network_state.waiting_passengers().values(),
                                               key=lambda passenger_group: passenger_group.priority / (
                                                   self.all_shortest_paths[train.position][0][passenger_group.position] + 1),
                                               reverse=True)
        for passenger_group in passengers_sorted_by_priority:
            if not passenger_group.is_assigned and passenger_group.group_size <= train.capacity:
                # Select matching passenger group for train
                train.assigned_passenger_group = passenger_group
                passenger_group.is_assigned = True
                return passenger_group
        return None


    def depart_train(self, round_action: RoundAction, train: MultiTrain, line: MultiLine):
        next_station_id = line.end if train.position == line.start else line.start
        train.station_state = TrainState.DEPARTING
        train.path = train.path[1:]
        round_action.train_departs[train.name] = line.name
        self.network_state.stations[next_station_id].locks.add(train.name)

    def swap_for_all_arriving(self, round_action: RoundAction):
        for train in [train for train in self.network_state.trains.values() if
                      train.station_state == TrainState.ARRIVING]:
            next_station = self.network_state.stations[train.next_station]
            if len(next_station.locks) + len(next_station.trains) > next_station.capacity:
                for leaving_train_name in next_station.trains:
                    leaving_train = self.network_state.trains[leaving_train_name]
                    if len(leaving_train.path) <= 1 or leaving_train.station_state == TrainState.DEPARTING:
                        continue
                    next_line_id = self.network_graph.edges[leaving_train.position, leaving_train.path[1]]['name']
                    next_line = self.network_state.lines[next_line_id]
                    if leaving_train.station_state == TrainState.WAITING_FOR_SWAP and next_line.name == train.position:
                        self.depart_train(round_action, leaving_train, next_line)
                        break
    def update_all_train_routes(self, round_action: RoundAction):
        # Check if all passengers have reached their destination to prevent keep planing unnecessary tours
        passengers_not_reached_destination = [passenger_group.name for passenger_group in self.network_state.passenger_groups.values() if passenger_group.destination != passenger_group.position]
        if len(passengers_not_reached_destination) == 0:
            return
        for train in sorted(self.network_state.trains.values(), key=lambda train: train.speed, reverse=True):
            if len(train.path) == 0 and train.position is not None:
                if train.assigned_passenger_group is None:
                    self.reserve_next_passenger(train, round_action)
                    if train.assigned_passenger_group is not None:
                        if train.assigned_passenger_group.position == train.name:
                            self.update_route_for_boarded_passenger(train)
                        else:
                            self.update_route_to_assigned_passenger(train)
                    else:
                        self.update_route_prevent_full_stations(train, round_action)
                else:
                    self.update_route_for_boarded_passenger(train)

    def board_all_reserved_passengers_for_train(self, train: MultiTrain, round_action: RoundAction):
        assert len(train.path) == 0
        assert train.position in [train.assigned_passenger_group.position, train.assigned_passenger_group.destination]
        train.path = []

        # Board passengers with reserved capacity
        if train.assigned_passenger_group.position == train.position:
            round_action.passenger_boards[train.assigned_passenger_group.name] = train.name
        # Detrain passengers that have reached their destination
        else:
            round_action.passenger_detrains.append(train.assigned_passenger_group.name)
            train.assigned_passenger_group = None
        train.station_state = TrainState.BOARDING

    def detrain_additional_passengers(self, round_action: RoundAction):
        for train in [train for train in self.network_state.trains.values() if
                      train.position_type == TrainPositionType.STATION and train.station_state != TrainState.DEPARTING]:
            # Detrain
            for passenger_group_name in train.passenger_groups:
                passenger_group = self.network_state.passenger_groups[passenger_group_name]
                if passenger_group.name not in round_action.passenger_detrains and (
                    passenger_group.destination == train.position or self.all_shortest_paths[train.position][1][
                                                                         passenger_group.destination][
                                                                     0:2] != train.path[0:2]):
                    assert passenger_group != train.assigned_passenger_group
                    round_action.passenger_detrains.append(passenger_group.name)
                    train.station_state = TrainState.BOARDING

    def board_additional_passengers(self, round_action: RoundAction):
        for train in [train for train in self.network_state.trains.values() if
                      train.position_type == TrainPositionType.STATION and train.station_state != TrainState.DEPARTING]:
            current_station = self.network_state.stations[train.position]
            # Board
            if len(current_station.passenger_groups) != 0:
                free_capacity = self.calculate_free_capacity(train)
                for new_passenger_group_name in current_station.passenger_groups:
                    new_passenger_group = self.network_state.passenger_groups[new_passenger_group_name]
                    # Check if there is no other train that wants the passenger, there is enough space and it is the right direction
                    if not new_passenger_group.is_assigned and free_capacity >= new_passenger_group.group_size and not new_passenger_group.is_destination_reached() and \
                        self.all_shortest_paths[train.position][1][new_passenger_group.destination][0:2] == train.path[0:2] and new_passenger_group_name not in round_action.passenger_boards:
                        train.station_state = TrainState.BOARDING
                        round_action.passenger_boards[new_passenger_group.name] = train.name
                        free_capacity -= new_passenger_group.group_size

    def release_all_station_locks(self):
        for train in self.network_state.trains.values():
            if train.position_type == TrainPositionType.STATION:
                self.network_state.stations[train.position].locks.discard(train.name)

    def depart_all_trains(self, round_action: RoundAction):
        for train in [train for train in self.network_state.trains.values() if train.position_type == TrainPositionType.STATION and len(train.path) > 1 and train.station_state not in [TrainState.BOARDING, TrainState.DEPARTING]]:
            assert train.position == train.path[0]
            next_line = self.network_state.lines[self.network_graph.edges[train.position, train.path[1]]['name']]

            if (next_line.reserved_capacity + len(next_line.trains) - next_line.capacity) < 0 or next_line.length <= train.speed:
                next_station = self.network_state.stations[train.path[1]]
                if len(next_station.locks) + len(next_station.trains) < next_station.capacity or train.name in next_station.locks:
                    if next_line.length > train.speed:
                        next_line.reserved_capacity += 1
                    self.depart_train(round_action, train, next_line)
                else:
                    train.station_state = TrainState.BLOCKED_STATION
            else:
                if train.station_state != TrainState.WAITING_FOR_SWAP:
                    train.station_state = TrainState.BLOCKED_LINE

    def resolve_blocked_station_leaving(self, round_action: RoundAction):
        leaving_trains = {train for train in self.network_state.trains.values() if
                          train.station_state == TrainState.DEPARTING}
        while len(leaving_trains) != 0:
            leaving_train = leaving_trains.pop()
            for blocked_train in [train for train in self.network_state.trains.values() if
                                  train.station_state == TrainState.BLOCKED_STATION]:
                if blocked_train.path[1] == leaving_train.position:
                    next_line_id = self.network_graph.edges[blocked_train.position, leaving_train.position]['name']
                    next_line = self.network_state.lines[next_line_id]
                    next_station = self.network_state.stations[leaving_train.position]
                    if not len(next_station.trains) + len(next_station.locks) - 1 < next_station.capacity:
                        continue
                    if blocked_train.speed >= next_line.length:
                        pass
                    elif next_line.reserved_capacity + len(next_line.trains) < next_line.capacity:
                        next_line.reserved_capacity += 1
                    else:
                        continue
                    self.depart_train(round_action, blocked_train, next_line)
                    leaving_trains.add(blocked_train)
                    break

    def resolve_blocked_station_swap(self, round_action: RoundAction):
        processed = set()
        for pair in combinations([train.name for train in self.network_state.trains.values() if
                                  train.station_state == TrainState.BLOCKED_STATION], 2):
            train1 = self.network_state.trains[pair[0]]
            train2 = self.network_state.trains[pair[1]]
            if train1.position == train2.position or train1.name in processed or train2.name in processed:
                continue
            set1 = set(train1.path[0:2])
            set2 = set(train2.path[0:2])
            common_path = set1.intersection(set2)
            if len(common_path) == 2:
                next_line_id = self.network_graph.edges[train1.position, train1.path[1]]['name']
                next_line = self.network_state.lines[next_line_id]
                if next_line.length <= train1.speed and next_line.length <= train2.speed:
                    pass
                elif (next_line.length > train1.speed and next_line.length <= train2.speed) or (
                    next_line.length <= train1.speed and next_line.length > train2.speed):
                    if next_line.reserved_capacity + len(next_line.trains) - next_line.capacity >= 0:
                        continue
                    next_line.reserved_capacity += 1
                else:
                    if next_line.reserved_capacity + len(next_line.trains) > next_line.capacity - 2:
                        if next_line.reserved_capacity + len(next_line.trains) > next_line.capacity - 1:
                            continue
                        else:
                            if train1.speed > train2.speed:
                                self.depart_train(round_action, train1, next_line)
                                processed.add(train1.name)
                                processed.add(train2.name)
                                train2.station_state = TrainState.WAITING_FOR_SWAP
                                self.network_state.stations[train1.position].locks.add(train2.name)
                            else:
                                self.depart_train(round_action, train2, next_line)
                                processed.add(train1.name)
                                processed.add(train2.name)
                                train1.station_state = TrainState.WAITING_FOR_SWAP
                                self.network_state.stations[train2.position].locks.add(train1.name)
                            next_line.reserved_capacity += 1
                            continue
                    next_line.reserved_capacity += 2
                self.depart_train(round_action, train1, next_line)
                self.depart_train(round_action, train2, next_line)
                processed.add(train1.name)
                processed.add(train2.name)

    def process_trains_at_final_destination(self, round_action: RoundAction):
        for train in self.network_state.trains.values():
            # Check if train is arriving at final destination
            if len(train.path) == 1 and train.position_type == TrainPositionType.STATION and train.station_state != TrainState.DEPARTING:
                train.path = []
                # Check if train is transporting (or will transporting) a passenger_group
                if train.assigned_passenger_group is not None:
                    self.board_all_reserved_passengers_for_train(train, round_action)

    def resolve_blocked_trains_without_passenger(self):
        # Reset all paths of trains without an assigned passenger, so we can check in the next round if there is a better path
        for train in [train for train in self.network_state.trains.values() if
                      train.assigned_passenger_group is None and train.station_state in [TrainState.BLOCKED_STATION,
                                                                                         TrainState.BLOCKED_LINE]]:
            train.path = []

    def schedule(self) -> Schedule:
        self.compute_priorities()

        original_state = deepcopy(self.network_state)

        # Create round action for zero Round
        round_action = self.place_wildcard_trains()

        actions = dict()
        round_id = 0
        actions[round_id] = round_action
        self.network_state.apply(round_action)

        self.update_all_train_routes(round_action)

        # Game loop, till there are no more passengers to transport
        while True:
            round_id += 1
            print(f"Processing round {round_id}")
            for line in self.network_state.lines.values():
                line.reserved_capacity = 0
            for train in self.network_state.trains.values():
                if train.position_type == TrainPositionType.LINE and train.speed + train.line_progress >= \
                    self.network_state.lines[train.position].length:
                    train.station_state = TrainState.ARRIVING
                elif train.station_state != TrainState.WAITING_FOR_SWAP:
                    train.station_state = TrainState.UNUSED
            round_action = RoundAction()

            self.release_all_station_locks()
            self.swap_for_all_arriving(round_action)
            self.process_trains_at_final_destination(round_action)
            self.update_all_train_routes(round_action)
            self.process_trains_at_final_destination(round_action)
            self.update_all_train_routes(round_action)
            self.detrain_additional_passengers(round_action)
            self.board_additional_passengers(round_action)
            self.depart_all_trains(round_action)
            self.resolve_blocked_station_swap(round_action)
            self.resolve_blocked_station_leaving(round_action)
            self.resolve_blocked_trains_without_passenger()

            actions[round_id] = round_action
            self.network_state.apply(round_action)
            if self.network_state.is_finished():
                break
        schedule = Schedule.from_dict(actions)

        # sanity check
        schedule_is_valid, error_round_id = schedule.is_valid(original_state)
        if not schedule_is_valid:
            raise Exception(f"invalid state at round {error_round_id}")

        return schedule
