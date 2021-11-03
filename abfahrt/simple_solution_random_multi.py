import logging
from copy import deepcopy
import time
from dataclasses import field, dataclass, asdict
from itertools import combinations
from random import shuffle, randint, random, seed
from typing import Dict, List, Set, Tuple, Optional

import networkx as nx
from networkx.algorithms import all_pairs_dijkstra

from abfahrt.simple_solution_multiple_trains import TrainState, MultiTrain, MultiLine, MultiStation, MultiPassengerGroup, MultiNetworkState, multify_network, SimpleSolverMultipleTrains
from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction, Train, PassengerGroup, Station, Line, \
    PassengerGroupPositionType


class SimpleSolverMultipleTrainsRandom(SimpleSolverMultipleTrains):
    def __init__(self, network_state: NetworkState, network_graph: nx.Graph):
        super().__init__(network_state, network_graph)
        new_seed = time.time()
        print(f'SEED: {new_seed}')
        seed(new_seed)

    def place_wildcard_trains(self) -> RoundAction:
        new_round_action = RoundAction()

        wildcard_trains = list(filter(lambda train: train.position_type == TrainPositionType.NOT_STARTED,
                                      self.network_state.trains.values()))
        shuffle(wildcard_trains)

        station_space_left = dict()
        for station_id, station in self.network_state.stations.items():
            station_space_left[station_id] = station.capacity - len(station.trains)

        station_keys = list(self.network_state.stations.keys())
        shuffle(station_keys)
        wildcard_train_index = 0
        placed = len(wildcard_trains) == 0
        while True:
            for key in station_keys:
                station = self.network_state.stations[key]

                if station_space_left[station.name] <= 0:
                    continue
                if wildcard_train_index >= len(wildcard_trains):
                    break

                for _ in range(randint(0, station_space_left[station.name])):
                    if wildcard_train_index >= len(wildcard_trains):
                        break
                    placed = True
                    new_round_action.train_starts[wildcard_trains[wildcard_train_index].name] = station.name
                    wildcard_train_index += 1
                    station_space_left[station.name] -= 1
            if not (all([train.position_type == TrainPositionType.NOT_STARTED for train in
                 self.network_state.trains.values()]) and not placed):
                break

        return new_round_action

    def reserve_next_passenger(self, train: MultiTrain, round_action: RoundAction, mutate: bool = False) -> Optional[MultiPassengerGroup]:
        assert train.assigned_passenger_group is None
        assert train.position_type == TrainPositionType.STATION

        if len([passenger_group for passenger_group in train.passenger_groups if
                passenger_group not in round_action.passenger_detrains and self.network_state.passenger_groups[
                    passenger_group].destination != train.position]) != 0:
            train.assigned_passenger_group = max(
                [self.network_state.passenger_groups[passenger_group] for passenger_group in train.passenger_groups if
                 passenger_group not in round_action.passenger_detrains and self.network_state.passenger_groups[
                     passenger_group].destination != train.position],
                key=lambda passenger_group: passenger_group.priority)
            train.assigned_passenger_group.is_assigned = False
            return train.assigned_passenger_group
        waiting_passengers = [passenger for passenger in self.network_state.waiting_passengers().values() if not passenger.is_assigned ]
        if mutate:
            shuffle(waiting_passengers)
        else:
            waiting_passengers = sorted(waiting_passengers,
                                               key=lambda passenger_group: passenger_group.priority / (
                                                   self.all_shortest_paths[train.position][0][
                                                       passenger_group.position] + 1),
                                               reverse=True)

        for passenger_group in waiting_passengers:
            if not passenger_group.is_assigned and passenger_group.group_size <= train.capacity:
                # Select matching passenger group for train
                train.assigned_passenger_group = passenger_group
                passenger_group.is_assigned = True
                return passenger_group
        return None

    def update_all_train_routes(self, round_action: RoundAction):
        # Check if all passengers have reached their destination to prevent keep planing unnecessary tours
        passengers_not_reached_destination = [passenger_group.name for passenger_group in
                                              self.network_state.passenger_groups.values() if
                                              passenger_group.destination != passenger_group.position]
        if len(passengers_not_reached_destination) == 0:
            return
        for train in sorted(self.network_state.trains.values(), key=lambda train: train.speed, reverse=True):
            if len(train.path) == 0 and train.position_type == TrainPositionType.STATION:
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

    def mutate_train_routes(self, round_action: RoundAction):
        # Check if all passengers have reached their destination to prevent keep planing unnecessary tours
        passengers_not_reached_destination = [passenger_group.name for passenger_group in
                                              self.network_state.passenger_groups.values() if
                                              passenger_group.destination != passenger_group.position]
        if len(passengers_not_reached_destination) == 0:
            return
        shuffled_trains = list(self.network_state.trains.values())
        shuffle(shuffled_trains)
        for train in shuffled_trains:
            if len(train.path) == 0 and train.position_type == TrainPositionType.STATION:
                if train.assigned_passenger_group is None:
                    self.reserve_next_passenger(train, round_action, True)
                    if train.assigned_passenger_group is not None:
                        if train.assigned_passenger_group.position == train.name:
                            self.update_route_for_boarded_passenger(train)
                        else:
                            self.update_route_to_assigned_passenger(train)
                    else:
                        self.update_route_prevent_full_stations(train, round_action)
                else:
                    self.update_route_for_boarded_passenger(train)

    def get_init_generation(self, original_state, gen_size):
        generation = []
        for i in range(gen_size):
            self.network_state = deepcopy(original_state)

            # Create round action for zero Round
            if i == 0:
                round_action = super().place_wildcard_trains()
            else:
                round_action = self.place_wildcard_trains()

            actions = dict()
            round_id = 0
            if round_action.is_zero_round():
                actions[round_id] = round_action
                self.network_state.apply(round_action)

            self.update_all_train_routes(round_action)

            mutate_amount = 0
            # Game loop, till there are no more passengers to transport
            broken_schedule = False
            while True:
                round_id += 1
                # print(f"Processing round {round_id}")
                round_action = RoundAction()
                for line in self.network_state.lines.values():
                    line.reserved_capacity = 0
                for train in self.network_state.trains.values():
                    if train.position_type == TrainPositionType.LINE and train.speed + train.line_progress >= \
                        self.network_state.lines[train.position].length:
                        train.station_state = TrainState.ARRIVING
                    elif train.station_state != TrainState.WAITING_FOR_SWAP:
                        train.station_state = TrainState.UNUSED

                if random() < 0.2 and i != 0:
                    mutate_amount = randint(1, 5)
                self.release_all_station_locks()
                self.swap_for_all_arriving(round_action)
                self.process_trains_at_final_destination(round_action)

                if mutate_amount > 0:
                    self.mutate_train_routes(round_action)
                    # TODO: mutate board and detrain additional passengers
                    mutate_amount -= 1
                else:
                    self.update_all_train_routes(round_action)
                self.process_trains_at_final_destination(round_action)
                self.update_all_train_routes(round_action)
                if mutate_amount == 0:
                    self.board_additional_passengers(round_action)
                self.detrain_additional_passengers(round_action)
                self.depart_all_trains(round_action)
                self.resolve_blocked_station_swap(round_action)
                self.resolve_blocked_station_leaving(round_action)
                self.resolve_blocked_trains_without_passenger()

                actions[round_id] = round_action
                self.network_state.apply(round_action)
                if round_id > 1200: # TODO: resolve deadlocks smarter
                    broken_schedule = True
                    break
                if self.network_state.is_finished():
                    break
            if not broken_schedule:
                total_delay = self.network_state.total_delay()
                print(total_delay, i)
                generation.append((Schedule.from_dict(actions), total_delay))
        return generation

    def mutate(self, original_schedule, original_state):
        self.network_state = deepcopy(original_state)

        actions = dict()
        round_id = 0
        # wildcard trains
        round_action = original_schedule.actions[round_id]  # TODO: copy needed?
        if round_action.is_zero_round():
            actions[round_id] = round_action
            self.network_state.apply(round_action)

        for _ in range(randint(1, len(original_schedule.actions))):
            round_id += 1
            round_action = original_schedule.actions[round_id]
            actions[round_id] = round_action
            self.network_state.apply(round_action)

        self.update_all_train_routes(round_action)

        mutate_amount = 0
        # Game loop, till there are no more passengers to transport
        while True:
            round_id += 1
            # print(f"Processing round {round_id}")
            round_action = RoundAction()
            for line in self.network_state.lines.values():
                line.reserved_capacity = 0
            for train in self.network_state.trains.values():
                if train.position_type == TrainPositionType.LINE and train.speed + train.line_progress >= \
                    self.network_state.lines[train.position].length:
                    train.station_state = TrainState.ARRIVING
                elif train.station_state != TrainState.WAITING_FOR_SWAP:
                    train.station_state = TrainState.UNUSED

            #if random() < 0.2:
                #mutate_amount = randint(1, max(len(original_schedule.actions) // 10, 2))
            self.release_all_station_locks()
            self.swap_for_all_arriving(round_action)
            self.process_trains_at_final_destination(round_action)

            if random() < 0.4: #mutate_amount > 0:
                self.mutate_train_routes(round_action)
                # TODO: mutate board and detrain additional passengers
                #mutate_amount -= 1
            else:
                self.update_all_train_routes(round_action)

            self.process_trains_at_final_destination(round_action)
            self.update_all_train_routes(round_action)
            if random() < 0.3: # mutate_amount == 0:
                self.board_additional_passengers(round_action)
            self.detrain_additional_passengers(round_action)
            self.depart_all_trains(round_action)
            self.resolve_blocked_station_swap(round_action)
            self.resolve_blocked_station_leaving(round_action)
            self.resolve_blocked_trains_without_passenger()

            actions[round_id] = round_action
            self.network_state.apply(round_action)
            if not self.network_state.is_valid():
                print("Invalid State occured")
                return
            if round_id > 1000:  # TODO: make deadlock better
                print("Ran into deadlock")
                return  # error in mutation
            if self.network_state.is_finished():
                break
        sced = Schedule.from_dict(actions)
        return sced, self.network_state.total_delay()

    def schedule(self) -> Schedule:
        self.compute_priorities()
        original_state = deepcopy(self.network_state)

        generation_size = 100
        iterations = 2
        generation = self.get_init_generation(original_state, generation_size)

        for generation_id in range(1, 31):
            new_generation = []
            for individual_id in range(len(generation)):
                new_individual = self.mutate(generation[individual_id][0], original_state)
                if new_individual is None:
                    continue
                else:
                    new_generation.append(new_individual)

            generation.extend(new_generation)
            generation = sorted(generation, key=lambda individual: individual[1])[:generation_size]
            best_schedule = generation[0][0]
            min_total_delay = generation[0][1]
            print(f'best score at gen #{generation_id}: {min_total_delay}')

        # sanity check
        schedule_is_valid, error_round_id = best_schedule.is_valid(original_state)
        if not schedule_is_valid:
            raise Exception(f"invalid state at round {error_round_id}")
        # check if finished
        state = deepcopy(original_state)
        state.apply_all(best_schedule)
        if not state.is_finished():
            raise Exception('game state is not finished - problem with schedule')
        return best_schedule
