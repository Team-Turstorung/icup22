from dataclasses import field, dataclass, asdict
import enum
import logging
from typing import Dict, List, Set, Tuple
from itertools import combinations

import networkx as nx
from networkx.algorithms import all_pairs_dijkstra

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction, Train, PassengerGroup, Station, Line, PassengerGroupPositionType


class TrainState(enum.Enum):
    BOARDING = 0
    BLOCKED_STATION = 1
    DEPARTING = 2
    UNUSED = 3
    BLOCKED_LINE = 4
    ARRIVING = 5
    READY_TO_LEAVE = 6
    ON_LINE = 7

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

@dataclass(unsafe_hash=True)
class MultiTrain(Train):
    path: List[str] = field(default_factory=list, compare=False)
    reserved_capacity: int = 0
    assigned_passenger_groups: Set[str] = field(default_factory=set, compare=False)
    station_state: TrainState = TrainState.UNUSED

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
        passenger_list = sorted(self.network_state.passenger_groups.values(), reverse=True, key=lambda passenger: passenger.priority)

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

    def navigate_train(self, train: MultiTrain, round_action: RoundAction):
        if train.position_type == TrainPositionType.LINE or len(train.path) == 0:
            return
        current_station = self.network_state.stations[train.position]
        current_station.locks.discard(train.name)

        # Detrain passengers
        for passenger_group_name in train.passenger_groups:
            passenger_group = self.network_state.passenger_groups[passenger_group_name]
            if passenger_group.destination == train.position or self.all_shortest_paths[train.position][1][passenger_group.destination][0:2] != train.path[0:2]:
                round_action.passenger_detrains.append(passenger_group.name)
                train.station_state = TrainState.BOARDING

        print(train.station_state, train.path)
        # We reached our current destination
        if len(train.path) == 1:
            train.path = []
            # Board passengers with reserved capacity
            if train.reserved_capacity != 0:
                passenger_groups = [self.network_state.passenger_groups[passenger_group_id] for passenger_group_id in current_station.passenger_groups]
                if len(passenger_groups) > 0:
                    for passenger_group in sorted(passenger_groups, key=lambda group: group.priority):
                        if passenger_group.name in train.assigned_passenger_groups:
                            train.assigned_passenger_groups.remove(passenger_group.name)
                            passenger_group.is_assigned = False
                            round_action.passenger_boards[passenger_group.name] = train.name
                            #train.path = self.all_shortest_paths[train.position][1][passenger_group.destination]
                            train.station_state = TrainState.BOARDING
                            break
            elif train.station_state != TrainState.BOARDING:
                return

        # Check if we can take a new passenger with us (without reservation)
        if len(current_station.passenger_groups) != 0:
            free_capacity = self.calculate_free_capacity(train)
            for new_passenger_group_name in current_station.passenger_groups:
                new_passenger_group = self.network_state.passenger_groups[new_passenger_group_name]
                # Check if there is no other train that wants the passenger, there is enough space and it is the right direction
                if not new_passenger_group.is_assigned and free_capacity >= new_passenger_group.group_size and not new_passenger_group.is_destination_reached():
                    if self.all_shortest_paths[train.position][1][new_passenger_group.destination][0:2] == train.path[0:2]:
                        # Because we board a new passenger we can't go on this round
                        if new_passenger_group.name in round_action.passenger_boards:
                            continue
                        train.station_state = TrainState.BOARDING
                        round_action.passenger_boards[new_passenger_group.name] = train.name
                        free_capacity -= new_passenger_group.group_size
        if train.station_state == TrainState.BOARDING:
            return

        assert len(train.path) == 0

        # go to next station in path
        next_line_id = self.network_graph.edges[train.position, train.path[1]]['name']
        next_line = self.network_state.lines[next_line_id]
        if (next_line.reserved_capacity + len(next_line.trains) - next_line.capacity) < 0 or next_line.length <= train.speed:
            next_station = self.network_state.stations[train.path[1]]
            if len(next_station.locks) + len(next_station.trains) < next_station.capacity or train.name in next_station.locks:
                if next_line.length > train.speed:
                    next_line.reserved_capacity += 1
                self.depart_train(round_action, train, next_line)
            else:
                train.station_state = TrainState.BLOCKED_STATION
        else:
            if train.station_state != TrainState.READY_TO_LEAVE:
                train.station_state = TrainState.BLOCKED_LINE

    def calculate_free_capacity(self, train: MultiTrain) -> int:
        # Get the capacity that is left on a train, to decide if we want new passengers.
        occupied_space = 0
        for passenger_group_name in train.passenger_groups:
            passenger_group = self.network_state.passenger_groups[passenger_group_name]
            occupied_space += passenger_group.group_size
        return train.capacity - occupied_space - train.reserved_capacity

    def plan_train(self, train: MultiTrain):
        if not train.position:
            return
        if len(train.passenger_groups) == 0:
            # go to suitable passenger group
            passengers_sorted_by_priority = sorted(self.network_state.waiting_passengers().values(),
                                                   key=lambda passenger_group: passenger_group.priority / (
                                                       self.all_shortest_paths[train.position][0][passenger_group.position] + 1),
                                                   reverse=True)

            for passenger_group in passengers_sorted_by_priority:
                if not passenger_group.is_assigned and passenger_group.group_size <= train.capacity:
                    # Select matching passenger group for train
                    train.assigned_passenger_groups.add(passenger_group.name)
                    passenger_group.is_assigned = True
                    train.reserved_capacity = passenger_group.group_size
                    return self.all_shortest_paths[train.position][1][passenger_group.position]

            if self.network_state.stations[train.position].is_full():
                for neighbor in self.network_graph.neighbors(train.position):
                    if not self.network_state.stations[neighbor].is_full():
                        return [train.position, neighbor]

        else:
            # go to destination of passenger group with highest priority
            passenger_group = self.network_state.passenger_groups[train.passenger_groups[0]]
            for passenger_group_name in train.passenger_groups[1:]:
                current_passenger_group = self.network_state.passenger_groups[passenger_group_name]
                if current_passenger_group.priority > passenger_group.priority:
                    passenger_group = current_passenger_group
            train.reserved_capacity = 0
            return self.all_shortest_paths[train.position][1][passenger_group.destination]

    def depart_train(self, round_action: RoundAction, train: MultiTrain, line: MultiLine):
        next_station_id = line.end if train.position == line.start else line.start
        train.station_state = TrainState.DEPARTING
        train.path = train.path[1:]
        round_action.train_departs[train.name] = line.name
        self.network_state.stations[next_station_id].locks.add(train.name)

    def depart_for_all_arriving(self, round_action: RoundAction):
        for train in [train for train in self.network_state.trains.values() if train.station_state == TrainState.ARRIVING]:
            next_station = self.network_state.stations[train.next_station]
            if len(next_station.locks) + len(next_station.trains) > next_station.capacity:
                for leaving_train_name in next_station.trains:
                    leaving_train = self.network_state.trains[leaving_train_name]
                    if len(leaving_train.path) <= 1:
                        continue
                    next_line_id = self.network_graph.edges[leaving_train.position, leaving_train.path[1]]['name']
                    next_line = self.network_state.lines[next_line_id]
                    if leaving_train.station_state == TrainState.READY_TO_LEAVE and next_line.name == train.position:
                        self.depart_train(round_action, leaving_train, next_line)
                        break

    def plan_all_trains(self):
        for train in sorted(self.network_state.trains.values(), key=lambda train: train.speed, reverse=True):
            if len(train.path) != 0:
                continue
            path = self.plan_train(train)
            if path is not None:
                train.path = path

    def navigate_all_trains(self, round_action: RoundAction):
        for train in sorted([train for train in self.network_state.trains.values() if not train.station_state == TrainState.DEPARTING], key=lambda train: train.speed, reverse=True):
            if len(train.path) != 0:
                self.navigate_train(train, round_action)

    def resolve_blocked_station_leaving(self, round_action: RoundAction):
        leaving_trains = {train for train in self.network_state.trains.values() if train.station_state == TrainState.DEPARTING}
        while len(leaving_trains) != 0:
            leaving_train = leaving_trains.pop()
            for blocked_train in [train for train in self.network_state.trains.values() if train.station_state == TrainState.BLOCKED_STATION]:
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
        for pair in combinations([train.name for train in self.network_state.trains.values() if train.station_state == TrainState.BLOCKED_STATION], 2):
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
                elif (next_line.length > train1.speed and next_line.length <= train2.speed) or (next_line.length <= train1.speed and next_line.length > train2.speed):
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
                                train2.station_state = TrainState.READY_TO_LEAVE
                                self.network_state.stations[train1.position].locks.add(train2.name)
                            else:
                                self.depart_train(round_action, train2, next_line)
                                processed.add(train1.name)
                                processed.add(train2.name)
                                train1.station_state = TrainState.READY_TO_LEAVE
                                self.network_state.stations[train2.position].locks.add(train1.name)
                            next_line.reserved_capacity += 1
                            continue
                    next_line.reserved_capacity += 2
                self.depart_train(round_action, train1, next_line)
                self.depart_train(round_action, train2, next_line)
                processed.add(train1.name)
                processed.add(train2.name)

    def schedule(self) -> Schedule:
        self.compute_priorities()

        # Create round action for zero Round
        round_action = self.place_wildcard_trains()

        actions = dict()
        round_id = 0
        actions[round_id] = round_action
        self.network_state.apply(round_action)

        # Game loop, till there are no more passengers to transport
        while True:
            print(f"Processing round {round_id}")
            for line in self.network_state.lines.values():
                line.reserved_capacity = 0
            for train in self.network_state.trains.values():
                if train.position_type == TrainPositionType.LINE and train.speed + train.line_progress >= self.network_state.lines[train.position].length:
                    train.station_state = TrainState.ARRIVING
                elif train.station_state != TrainState.READY_TO_LEAVE:
                    train.station_state = TrainState.UNUSED
            round_action = RoundAction()
            round_id += 1

            self.plan_all_trains()
            self.depart_for_all_arriving(round_action)
            self.navigate_all_trains(round_action)
            self.resolve_blocked_station_swap(round_action)
            self.resolve_blocked_station_leaving(round_action)

            actions[round_id] = round_action
            self.network_state.apply(round_action)
            if not self.network_state.is_valid():
                raise Exception(f"invalid state at round {round_id}")
            if self.network_state.is_finished():
                break
        schedule = Schedule.from_dict(actions)
        return schedule
