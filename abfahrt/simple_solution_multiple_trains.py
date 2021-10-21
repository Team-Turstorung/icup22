import logging
from typing import Dict, List
from itertools import combinations

import networkx as nx
from networkx.algorithms import all_pairs_dijkstra

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction, Train, PassengerGroup


class SimplesSolverMultipleTrains(Solution):
    def schedule(self, network_state: NetworkState, network_graph: nx.Graph) -> Schedule:

        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)

        def get_all_shortest_paths(network_graph: nx.Graph) -> Dict[str, tuple]:
            shortest_paths = all_pairs_dijkstra(network_graph)
            all_shortest_paths = {}
            for path in shortest_paths:
                all_shortest_paths[path[0]] = path[1]
            return all_shortest_paths

        def place_wildcard_trains(network_state: NetworkState) -> RoundAction:

            new_round_action = RoundAction()

            wildcard_trains = filter(lambda train: train.position_type == TrainPositionType.NOT_STARTED,
                                     network_state.trains.values())
            sorted_wildcard_trains = sorted(wildcard_trains, key=lambda train: train.speed, reverse=True)
            passenger_list = sorted(passenger_priorities.items(), reverse=True, key=lambda item: item[1])

            station_space_left = dict()
            for station_id, station in network_state.stations.items():
                station_space_left[station_id] = station.capacity - len(station.trains)

            for passenger_group_name in passenger_list:
                passenger_group_name = passenger_group_name[0]
                current_passenger_group = network_state.passenger_groups[passenger_group_name]
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

        def compute_priorities(passenger_groups: List[PassengerGroup]) -> Dict[str, int]:
            priorities = dict()
            for passenger_group in passenger_groups:
                priorities[passenger_group.name] = all_shortest_paths[passenger_group.position][0][
                                                       passenger_group.destination] / (
                                                       passenger_group.time_remaining + 1) * passenger_group.group_size
            return priorities

        def navigate_train(train: Train, path: List[str]):
            if train.position_type == TrainPositionType.LINE or train.name not in on_tour:
                return
            # Check if there is a passenger detrained
            pause_train = False
            for passenger_group_name in train.passenger_groups:
                passenger_group = network_state.passenger_groups[passenger_group_name]
                if passenger_group.destination == train.position or all_shortest_paths[train.position][1][passenger_group.destination][0:2] != path[0:2]:
                    detrain_passengers.append(passenger_group.name)
                    pause_train = True

            locks[train.position].discard(train.name)
            # We reached our current destination
            if len(path) == 1:
                # We have to pause cause we need a new destination first TODO: Look for new destination if we don't board or detrain
                pause_train = True
                # If there is reserved capacity we want to pickup a new passenger
                if reserved_capacity[train.name] != 0:
                    # pick up
                    board_passengers.append(train)
                on_tour.remove(train.name)

            current_station = network_state.stations[train.position]
            # Check if we can take a new passenger with us
            if len(current_station.passenger_groups) != 0:
                free_capacity = calculate_free_capacity(train)
                for new_passenger_group_name in current_station.passenger_groups:
                    new_passenger_group = network_state.passenger_groups[new_passenger_group_name]
                    # Check if there is no other train that wants the passenger, there is enough space and it is the right direction
                    if new_passenger_group.name not in all_assigned_passenger_groups and free_capacity >= new_passenger_group.group_size and not new_passenger_group.is_destination_reached():
                        if all_shortest_paths[train.position][1][new_passenger_group.destination][0:2] == path[0:2]:
                            assigned_passenger_groups[train.name].add(new_passenger_group.name)
                            all_assigned_passenger_groups.add(new_passenger_group.name)
                            # Because we board a new passenger we can't go on this round
                            pause_train = True
                            board_passengers.append(train)
                            free_capacity -= new_passenger_group.group_size
            if pause_train:
                return

            # go to next station in path
            next_line_id = network_graph.edges[train.position, path[1]]['name']
            next_line = network_state.lines[next_line_id]
            if (line_usage.get(next_line_id, 0) + len(next_line.trains) - next_line.capacity) < 0 or next_line.length <= train.speed:
                next_station = network_state.stations[path[1]]
                if len(locks[next_station.name]) + len(next_station.trains) - next_station.capacity < 0:
                    if next_line.name in line_usage:
                        line_usage[next_line.name] += 1
                    else:
                        line_usage[next_line.name] = 1
                    locks[next_station.name].add(train.name)
                    round_action.train_departs[train.name] = next_line_id
                    train_paths[train.name] = path[1:]
                else:
                    blocked_trains.append(train.name)

        def calculate_free_capacity(train: Train) -> int:
            # Get the capacity that is left on a train, to decide if we want new passengers.
            occupied_space = 0
            for passenger_group_name in train.passenger_groups:
                passenger_group = network_state.passenger_groups[passenger_group_name]
                occupied_space += passenger_group.group_size
            return train.capacity - occupied_space - reserved_capacity[train.name]

        def plan_train(train: Train):
            if len(train.passenger_groups) == 0:
                # go to suitable passenger group
                passengers_sorted_by_priority = sorted(network_state.waiting_passengers().values(),
                                                       key=lambda passenger_group: passenger_priorities[passenger_group.name] / (
                                                           all_shortest_paths[train.position][0][passenger_group.position] + 1),
                                                       reverse=True)

                for passenger_group in passengers_sorted_by_priority:
                    if passenger_group.name not in all_assigned_passenger_groups and passenger_group.group_size <= train.capacity:
                        # Select matching passenger group for train
                        assigned_passenger_groups[train.name].add(passenger_group.name)
                        all_assigned_passenger_groups.add(passenger_group.name)
                        reserved_capacity[train.name] = passenger_group.group_size
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
                    if passenger_priorities[current_passenger_group.name] > passenger_priorities[passenger_group.name]:
                        passenger_group = current_passenger_group
                reserved_capacity[train.name] = 0
                return all_shortest_paths[train.position][1][passenger_group.destination]

        all_shortest_paths = get_all_shortest_paths(network_graph)
        passenger_priorities = compute_priorities(list(network_state.passenger_groups.values()))

        # Create round action for zero Round
        round_action = place_wildcard_trains(network_state)

        actions = dict()
        round_id = 0
        actions[round_id] = round_action
        network_state.apply(round_action)
        on_tour = set()
        train_paths = {}  # train: path
        board_passengers = []
        detrain_passengers = []
        locks = {name: set() for name in network_state.stations}
        all_assigned_passenger_groups = set()
        reserved_capacity = {train: 0 for train in network_state.trains}
        assigned_passenger_groups = {train: set() for train in network_state.trains}

        # Game loop, till there are no more passengers to transport
        while True:
            print(f"Processing round {round_id}")
            blocked_trains = []
            line_usage = {}
            round_action = RoundAction()
            round_id += 1
            for train in sorted(network_state.trains.values(), key=lambda train: train.speed, reverse=True):
                if train.name in on_tour:
                    continue
                plan = plan_train(train)
                if plan is not None:
                    on_tour.add(train.name)
                    train_paths[train.name] = plan

            for train_name, path in train_paths.items():
                train = network_state.trains[train_name]
                navigate_train(train, path)

            processed = set()
            for pair in combinations(blocked_trains, 2):
                train1 = network_state.trains[pair[0]]
                train2 = network_state.trains[pair[1]]
                if train1.position == train2.position or train1.name in processed or train2.name in processed:
                    continue
                set1 = set(train_paths[train1.name][0:2])
                set2 = set(train_paths[train2.name][0:2])
                common_path = set1.intersection(set2)
                if len(common_path) == 2:
                    next_line_id = network_graph.edges[train1.position, train_paths[train1.name][1]]['name']
                    next_line = network_state.lines[next_line_id]
                    if ((line_usage.get(next_line_id, 0) + len(next_line.trains) - next_line.capacity) < 0 and (next_line.length <= train1.speed or next_line.length <= train2.speed)) or ((line_usage.get(next_line_id, 0) + len(next_line.trains) - next_line.capacity) < -1) or (next_line.length <= train1.speed and next_line.length <= train2.speed):
                        locks[train1.position].add(train2.name)
                        locks[train2.position].add(train1.name)
                        if next_line_id in line_usage:
                            line_usage[next_line_id] += 2
                        else:
                            line_usage[next_line_id] = 2
                        round_action.train_departs[train1.name] = next_line_id
                        round_action.train_departs[train2.name] = next_line_id
                        train_paths[train1.name] = train_paths[train1.name][1:]
                        train_paths[train2.name] = train_paths[train2.name][1:]
                        processed.add(train1.name)
                        processed.add(train2.name)

            for train in board_passengers:
                passenger_groups = network_state.stations[train.position].passenger_groups
                if not passenger_groups:
                    continue
                passenger_groups_sorted = sorted(passenger_groups, key=lambda name: passenger_priorities[name])
                for passenger_group_name in passenger_groups_sorted:
                    passenger_group = network_state.passenger_groups[passenger_group_name]
                    if not passenger_group.is_destination_reached() and passenger_group.name in assigned_passenger_groups[train.name]:
                        assigned_passenger_groups[train.name].remove(passenger_group.name)
                        all_assigned_passenger_groups.remove(passenger_group.name)
                        round_action.passenger_boards[passenger_group.name] = train.name
                        break

            board_passengers = []
            for passenger_group in detrain_passengers:
                round_action.passenger_detrains.append(passenger_group)
            detrain_passengers = []

            actions[round_id] = round_action
            network_state.apply(round_action)
            if not network_state.is_valid():
                raise Exception(f"invalid state at round {round_id}")
            if network_state.is_finished():
                break
        schedule = Schedule.from_dict(actions)
        return schedule
