import logging
from collections import defaultdict
from copy import deepcopy
from typing import Optional

from mip import Model, xsum, minimize, BINARY, OptimizationStatus, INTEGER

import networkx as nx

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, TrainPositionType, Schedule, RoundAction, PassengerGroupPositionType

PSEUDO_STATION_NAME = "pseudo"


class MipSolver(Solution):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def schedule(self, network_state: NetworkState, network_graph: nx.graph):
        max_rounds = 5
        solved_model: Optional[Model] = None
        solved_max_rounds: Optional[int] = None
        solved_position_variables = None
        while True:
            self.log.info('trying max_rounds = %s', max_rounds)
            m, position_variables = self.schedule_with_num_rounds(network_state, network_graph, max_rounds)
            if m.status == OptimizationStatus.OPTIMAL or m.status == OptimizationStatus.FEASIBLE:
                if m.status == OptimizationStatus.OPTIMAL:
                    self.log.info('optimal solution cost %s found', m.objective_value)
                else:
                    self.log.info('solution with objective value %s found, best possible: %s', m.objective_value, m.objective_bound)
                finished = solved_model is not None or m.objective_value == 0
                solved_model = m
                solved_max_rounds = max_rounds
                solved_position_variables = position_variables
                if finished:
                    break
            elif m.status == OptimizationStatus.INFEASIBLE:
                self.log.info("infeasable for max_rounds = %s", max_rounds)
            elif m.status == OptimizationStatus.NO_SOLUTION_FOUND:
                self.log.info("no solution found for max_rounds = %s", max_rounds)
            max_rounds = max_rounds * 5 // 4
        return self.solved_model_to_schedule(network_state, network_graph, solved_max_rounds, *solved_position_variables)

    def dicts_from_network(self, network_state: NetworkState):
        stations = {name: number for number, name in enumerate(network_state.stations.keys())}
        if PSEUDO_STATION_NAME not in stations:
            stations[PSEUDO_STATION_NAME] = max(stations.values()) + 1
        else:
            raise Exception("Please rename your pseudo station")
        passengers = {name: number for number, name in enumerate(network_state.passenger_groups.keys())}
        trains = {name: number for number, name in enumerate(network_state.trains.keys())}
        lines = {name: number for number, name in enumerate(network_state.lines.keys())}
        reverse_trains = {number: name for name, number in trains.items()}
        reverse_stations = {number: name for name, number in stations.items()}
        reverse_lines = {number: name for name, number in lines.items()}
        reverse_passengers = {number: name for name, number in passengers.items()}
        return stations, lines, trains, passengers, reverse_stations, reverse_lines, reverse_trains, reverse_passengers

    def schedule_with_num_rounds(self, network_state: NetworkState, network_graph: nx.graph, max_rounds: int):
        stations, lines, trains, passengers, reverse_stations, reverse_lines, _, _ = self.dicts_from_network(network_state)

        target_times = {passengers[passenger_id]: passenger.time_remaining for passenger_id, passenger in
                        network_state.passenger_groups.items()}
        group_sizes = {passengers[passenger_id]: passenger.group_size for passenger_id, passenger in
                        network_state.passenger_groups.items()}
        train_capacities = {trains[train_id]: train.capacity for train_id, train in
                       network_state.trains.items()}
        station_capacities = {stations[station_id]: station.capacity for station_id, station in
                            network_state.stations.items()}
        station_capacities[stations[PSEUDO_STATION_NAME]] = len(trains)
        line_capacities = {lines[line_id]: line.capacity for line_id, line in network_state.lines.items()}
        line_lengths = {lines[line_id]: line.length for line_id, line in network_state.lines.items()}
        train_initial_positions = {trains[train_id]: stations[train.position] if train.position_type == TrainPositionType.STATION else None for train_id, train in
                                   network_state.trains.items()}
        passenger_initial_positions = {passengers[passenger_id]: stations[passenger.position] for passenger_id, passenger in
                                       network_state.passenger_groups.items()}
        train_speeds = {trains[train_id]: train.speed for train_id, train in network_state.trains.items()}
        targets = {passengers[passenger_id]: stations[passenger.destination] for passenger_id, passenger in
                   network_state.passenger_groups.items()}

        m = Model()

        train_position_stations = [[[m.add_var(var_type=BINARY) for _ in trains]for _ in stations] for _ in range(max_rounds)]
        train_position_lines = [[[m.add_var(var_type=BINARY) for _ in trains] for _ in lines] for _ in range(max_rounds)]
        passenger_position_stations = [[[m.add_var(var_type=BINARY) for _ in passengers] for _ in stations] for _ in range(max_rounds)]
        passenger_position_trains = [[[m.add_var(var_type=BINARY) for _ in passengers] for _ in trains] for _ in range(max_rounds)]
        passenger_time = [[m.add_var(var_type=INTEGER) for _ in passengers] for _ in range(max_rounds)]
        passenger_delay = [m.add_var(var_type=INTEGER) for _ in passengers] # we use the implicit lower bound of 0 for the delay
        train_progress = [[[m.add_var() for _ in trains] for _ in lines] for _ in range(max_rounds)]
        train_destinations = [[[m.add_var(var_type=BINARY) for _ in stations] for _ in trains] for _ in range(max_rounds)]

        for round_vars in train_position_stations:
            sos = [[] for _ in trains]
            for station_vars in round_vars:
                for number, train_var in enumerate(station_vars):
                    sos[number] += [train_var]

            for train in sos:
                m.add_sos([(name, i) for i, name in enumerate(train)], 1)

        for round_vars in train_position_lines:
            sos = [[] for _ in trains]
            for station_vars in round_vars:
                for number, train_var in enumerate(station_vars):
                    sos[number] += [train_var]

            for train in sos:
                m.add_sos([(name, i) for i, name in enumerate(train)], 1)

        # Constraint: Set passenger delay to target time minus time elapsed
        for p in passengers.values():
            m += passenger_delay[p] == (passenger_time[max_rounds-1][p]-target_times[p])*group_sizes[p]

        # Constraint: Train cannot jump between lines
        for i in range(max_rounds-1):
            for l1 in lines.values():
                for l2 in lines.values():
                    if l1 == l2:
                        continue
                    for t in trains.values():
                        m += train_position_lines[i][l1][t] + train_position_lines[i + 1][l2][t] <= 1

        # Constraint: The train has one destination if it is on a line
        for i in range(max_rounds):
            for t in trains.values():
                m += xsum(train_destinations[i][t][s] for s in stations.values()) == xsum(train_position_lines[i][l][t] for l in lines.values())

        # Constraint: The destination cannot be last round's station. If a train has a destination, it keeps the destination or arrives at the station
        for i in range(max_rounds-1):
            for t in trains.values():
                for s in stations.values():
                    m += train_position_stations[i][s][t] + train_destinations[i+1][t][s] <= 1
                    m += train_destinations[i][t][s] <= train_destinations[i+1][t][s] + train_position_stations[i+1][s][t]

        # Constraint: When a train is in a station, it can go to station it can reach in one round and all neighbor lines
        for s1 in stations.values():
            for t in trains.values():
                possible_next_stations = [s1]
                possible_next_lines = []
                for s2 in stations.values():
                    if not network_graph.has_edge(reverse_stations[s1], reverse_stations[s2]):
                        continue
                    l = lines[network_graph[reverse_stations[s1]][reverse_stations[s2]]["name"]]
                    if train_speeds[t] >= line_lengths[l]:
                        possible_next_stations.append(s2)
                    else:
                        possible_next_lines.append(l)
                for i in range(max_rounds-1):
                    m += train_position_stations[i][s1][t] <= xsum(
                        train_position_stations[i + 1][s2][t] for s2 in possible_next_stations) + xsum(
                        train_position_lines[i + 1][l][t] for l in possible_next_lines)

        for i in range(max_rounds-1):
            for t in trains.values():
                for l in lines.values():
                    s1 = stations[network_state.lines[reverse_lines[l]].start]
                    s2 = stations[network_state.lines[reverse_lines[l]].end]
                    # Train arrives at station when last round's progress plus speed is greater than length
                    m += line_lengths[l] <= train_progress[i][l][t] + train_speeds[t] + line_lengths[l]*(2-train_position_stations[i+1][s2][t]-train_position_stations[i+1][s1][t]-train_position_lines[i][l][t])

                    # When on line, the train is on the line or in one of the stations in the next round
                    m += train_position_lines[i][l][t] <= train_position_stations[i+1][s1][t] + train_position_stations[i+1][s2][t] + train_position_lines[i+1][l][t]

        # Constraint: Trains ...
        for i in range(max_rounds-1):
            for l in lines.values():
                for t in trains.values():
                    # ... move can never travel further than their speed
                    m += train_progress[i+1][l][t] <= train_progress[i][l][t] + train_speeds[t]
                    # ... move with at least their speed when they are on the line in the next round
                    m += train_progress[i][l][t] + train_speeds[t] - (line_lengths[l]+train_speeds[t]) * (1-train_position_lines[i+1][l][t]) <= train_progress[i+1][l][t]
                    # ... can never be further on the line than the line is long. # TODO: is the <= instead of the < a problem here?
                    m += train_progress[i][l][t] <= line_lengths[l]*train_position_lines[i][l][t]

        # Constraint: When a passenger hops onto a train, the train must be in the corresponding station in two turns. The same is true when getting out.
        for i in range(max_rounds-1):
            for p in passengers.values():
                for t in trains.values():
                    for s in stations.values():
                        m += passenger_position_stations[i][s][p] + passenger_position_trains[i + 1][t][p] <= 1 + \
                             train_position_stations[i][s][t]
                        m += passenger_position_stations[i][s][p] + passenger_position_trains[i + 1][t][p] <= 1 + \
                             train_position_stations[i + 1][s][t]
                        m += passenger_position_trains[i][t][p] + passenger_position_stations[i + 1][s][p] <= 1 + \
                             train_position_stations[i][s][t]
                        m += passenger_position_trains[i][t][p] + passenger_position_stations[i + 1][s][p] <= 1 + \
                             train_position_stations[i + 1][s][t]

        # Constraint: Train capacities
        for i in range(max_rounds):
            for t in trains.values():
                m += xsum(group_sizes[p]*passenger_position_trains[i][t][p] for p in passengers.values()) <= train_capacities[t]

        # Constraint: Station capacities
        for i in range(max_rounds):
            for s in stations.values():
                m += xsum(train_position_stations[i][s][t] for t in trains.values()) <= station_capacities[s]

        # Constraint: Line capacities
        for i in range(max_rounds):
            for l in lines.values():
                m += xsum(train_position_lines[i][l][t] for t in trains.values()) <= line_capacities[l]

        # Constraint: Passengers cannot change between two different stations
        for i in range(max_rounds-1):
            for p in passengers.values():
                for s1 in stations.values():
                    for s2 in stations.values():
                        if s1 == s2:
                            continue
                        m += passenger_position_stations[i][s1][p] <= (1-passenger_position_stations[i+1][s2][p])

        # Constraint: Passengers cannot change between two different trains
        for i in range(max_rounds-1):
            for p in passengers.values():
                for t1 in trains.values():
                    for t2 in trains.values():
                        if t1 == t2:
                            continue
                        m += passenger_position_trains[i][t1][p] <= (1-passenger_position_trains[i+1][t2][p])

        # Constraint: Passengers do not move after destination reached
        for p in passengers.values():
            for i in range(max_rounds-1):
                m += passenger_position_stations[i+1][targets[p]][p] - passenger_position_stations[i][targets[p]][p] >= 0

        # Constraint: All trains are at one position at a time
        for t in trains.values():
            for i in range(max_rounds):
                m += xsum(train_position_stations[i][s][t] for s in stations.values()) + xsum(train_position_lines[i][l][t] for l in lines.values()) == 1

        # Constraint: All passengers are at one position at a time
        for p in passengers.values():
            for i in range(max_rounds):
                m += xsum(passenger_position_stations[i][s][p] for s in stations.values()) + xsum(passenger_position_trains[i][t][p] for t in trains.values()) == 1

        # Constraint: All trains are at their initial positions in the first round. If no position specified, use any station
        for t in trains.values():
            if train_initial_positions[t] is not None:
                m += train_position_stations[0][train_initial_positions[t]][t] == 1
            else:
                m += xsum(train_position_stations[0][s][t] for s in stations.values()) == 1

        # Constraint: All passenger groups are at their initial positions
        for p in passengers.values():
            m += passenger_position_stations[0][passenger_initial_positions[p]][p] == 1

        # Constraint: All passenger groups are at their final positions
        for p in passengers.values():
            m += passenger_position_stations[-1][targets[p]][p] == 1

        # Constraint: Time is updated if destination not reached
        for p in passengers.values():
            for i in range(max_rounds-1):
                m += passenger_time[i+1][p] == (passenger_time[i][p] + 1 - passenger_position_stations[i][targets[p]][p])

        # Constraint: Calculate passenger delay
        m.objective = minimize(xsum(
            passenger_delay[p] for p in passengers.values()
        ))

        self.log.info('%s vars, %s constraints and %s nzs', m.num_cols, m.num_rows, m.num_nz)
        m.optimize()
        return m, (train_position_stations, train_position_lines, passenger_position_stations, passenger_position_trains)

    def solved_model_to_schedule(self, network_state: NetworkState, network_graph: nx.graph, max_rounds: int, train_position_stations, train_position_lines, passenger_position_stations, passenger_position_trains):
        stations, lines, trains, passengers, reverse_stations, reverse_lines, reverse_trains, reverse_passengers = self.dicts_from_network(network_state)
        actions = defaultdict(RoundAction)
        current_state = deepcopy(network_state)
        for i in range(max_rounds):
            action = RoundAction()

            if i == 0:
                for t in trains.values():
                    if network_state.trains[reverse_trains[t]].position_type == TrainPositionType.NOT_STARTED:
                        for s in stations.values():
                            if s != stations[PSEUDO_STATION_NAME] and train_position_stations[i][s][t].x == 1:
                                action.train_starts[reverse_trains[t]] = reverse_stations[s]
            else:
                for t in trains.values():
                    current_train = current_state.trains[reverse_trains[t]]
                    for s in stations.values():
                        if train_position_stations[i][s][t].x == 1:
                            position_type = TrainPositionType.STATION
                            position = reverse_stations[s]
                    for l in lines.values():
                        if train_position_lines[i][l][t].x == 1:
                            position_type = TrainPositionType.LINE
                            position = reverse_lines[l]
                    if current_train.position_type == TrainPositionType.STATION and current_train.position != position:
                        if position_type == TrainPositionType.LINE:
                            action.train_departs[current_train.name] = position
                        elif position_type == TrainPositionType.STATION:
                            action.train_departs[current_train.name] = network_graph[current_train.position][position]["name"]

                for p in passengers.values():
                    current_passenger = current_state.passenger_groups[reverse_passengers[p]]
                    for s in stations.values():
                        if passenger_position_stations[i][s][p].x == 1:
                            position_type = PassengerGroupPositionType.STATION
                            position = reverse_stations[s]
                    for t in trains.values():
                        if passenger_position_trains[i][t][p].x == 1:
                            position_type = PassengerGroupPositionType.TRAIN
                            position = reverse_trains[t]
                    if current_passenger.position_type == PassengerGroupPositionType.TRAIN and position_type == PassengerGroupPositionType.STATION:
                        action.passenger_detrains.append(current_passenger.name)
                    elif current_passenger.position_type == PassengerGroupPositionType.STATION and position_type == PassengerGroupPositionType.TRAIN:
                        action.passenger_boards[current_passenger.name] = position

            if not action.is_empty():
                actions[i] = action
            if not (i == 0 and action.is_empty()):
                current_state.apply(action)
            if current_state.is_finished():
                break
        return Schedule(actions)
