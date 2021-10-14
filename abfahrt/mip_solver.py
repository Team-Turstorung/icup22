from collections import defaultdict
from copy import deepcopy

from mip import Model, xsum, minimize, BINARY, OptimizationStatus, INTEGER

import networkx as nx

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, TrainPositionType, Schedule, RoundAction, PassengerGroupPositionType


def mid(abfahrt_id):
    return int(abfahrt_id[1:])-1


def aid(mip_id, letter):
    return letter + str(mip_id+1)


class MipSolver(Solution):
    def schedule(self, network_state: NetworkState, network_graph: nx.graph):
        max_rounds = 15
        # constants
        max_line_length = max([line.length for line in network_state.lines.values()])
        stations = [mid(station_id) for station_id in network_state.stations.keys()]
        trains = [mid(train_id) for train_id in network_state.trains.keys()]
        passengers = [mid(passenger_id) for passenger_id in network_state.passenger_groups.keys()]
        lines = [mid(line_id) for line_id in network_state.lines.keys()]

        target_times = {mid(passenger_id): passenger.time_remaining for passenger_id, passenger in
                        network_state.passenger_groups.items()}
        group_sizes = {mid(passenger_id): passenger.group_size for passenger_id, passenger in
                        network_state.passenger_groups.items()}
        train_capacities = {mid(train_id): train.capacity for train_id, train in
                       network_state.trains.items()}
        station_capacities = {mid(station_id): station.capacity for station_id, station in
                            network_state.stations.items()}
        line_capacities = {mid(line_id): line.capacity for line_id, line in network_state.lines.items()}
        line_lengths = {mid(line_id): line.length for line_id, line in network_state.lines.items()}
        train_initial_positions = {mid(train_id): mid(train.position) if train.position_type == TrainPositionType.STATION else None for train_id, train in
                                   network_state.trains.items()}
        passenger_initial_positions = {mid(passenger_id): mid(passenger.position) for passenger_id, passenger in
                                       network_state.passenger_groups.items()}
        train_speeds = {mid(train_id): train.speed for train_id, train in network_state.trains.items()}
        targets = {mid(passenger_id): mid(passenger.destination) for passenger_id, passenger in
                   network_state.passenger_groups.items()}

        #adjacency_matrix = nx.adjacency_matrix(network_graph, weight=None)
        m = Model()

        train_position_stations = [[[m.add_var(var_type=BINARY) for _ in trains]for _ in stations] for _ in range(max_rounds)]
        train_position_lines = [[[m.add_var(var_type=BINARY) for _ in trains] for _ in lines] for _ in range(max_rounds)]
        passenger_position_stations = [[[m.add_var(var_type=BINARY) for _ in passengers] for _ in stations] for _ in range(max_rounds)]
        passenger_position_trains = [[[m.add_var(var_type=BINARY) for _ in passengers] for _ in trains] for _ in range(max_rounds)]
        passenger_time = [[m.add_var(var_type=INTEGER) for _ in passengers] for _ in range(max_rounds)]
        passenger_delay = [m.add_var(var_type=INTEGER) for _ in passengers] # we use the implicit lower bound of 0 for the delay
        train_progress = [[[m.add_var() for _ in trains] for _ in lines] for _ in range(max_rounds)]

        # Constraint: Set passenger delay to target time minus time elapsed
        for p in passengers:
            m += passenger_delay[p] == (passenger_time[max_rounds-1][p]-target_times[p])*group_sizes[p]

        # Constraint: Train cannot jump between lines
        for i in range(max_rounds-1):
            for l1 in lines:
                for l2 in lines:
                    if l1 == l2:
                        continue
                    for t in trains:
                        m += train_position_lines[i][l1][t] + train_position_lines[i + 1][l2][t] <= 1

        # Constraint: Trains cannot access stations or lines that are not connected to their current station or line
        for s1 in stations:
            for s2 in stations:
                if s1 == s2:
                    continue
                for t in trains:
                    if network_graph.has_edge(aid(s1, 'S'), aid(s2, 'S')):
                        l = mid(network_graph[aid(s1, 'S')][aid(s2, 'S')]["name"])
                        # Line is shorter than train is fast
                        if train_speeds[t] >= line_lengths[l]:
                            m += xsum(train_position_lines[i][l][t] for i in range(max_rounds)) == 0
                        # Line exists but is longer than train is fast
                        else:
                            for i in range(max_rounds-1):
                                # Train can never be at stations back-to-back
                                m += train_position_stations[i][s1][t] + train_position_stations[i+1][s2][t] <= 1
                                m += train_position_stations[i+1][s1][t] + train_position_stations[i][s2][t] <= 1

                                # Train arrives at station when last round's progress plus speed is greater than length
                                m += line_lengths[l] <= train_progress[i][l][t] + train_speeds[t] + max_line_length*(2-train_position_stations[i+1][s2][t]-train_position_stations[i+1][s1][t]-train_position_lines[i][l][t])

                                # Line is not reachable via any other station
                                for s3 in stations:
                                    if s3 in (s1, s2):
                                        continue
                                    m += train_position_stations[i][s3][t] + train_position_lines[i+1][l][t] <= 1
                                    m += train_position_stations[i+1][s3][t] + train_position_lines[i][l][t] <= 1
                    # Line does not exist, so train cannot change positions that way
                    else:
                        for i in range(max_rounds-1):
                            m += train_position_stations[i][s1][t] + train_position_stations[i+1][s2][t] <= 1
                            m += train_position_stations[i+1][s1][t] + train_position_stations[i][s2][t] <= 1

        # Constraint: Trains move with their speed, progress is zero when the train is not on the line, else bounded by line length
        for i in range(max_rounds-1):
            for l in lines:
                for t in trains:
                    m += train_progress[i+1][l][t] <= train_progress[i][l][t] + train_speeds[t]
                    m += train_progress[i][l][t] <= line_lengths[l]*train_position_lines[i][l][t]

        # Constraint: When a passenger hops onto a train, the train must be in the corresponding station in two turns. The same is true when getting out.
        for i in range(max_rounds-1):
            for p in passengers:
                for t in trains:
                    for s in stations:
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
            for t in trains:
                m += xsum(group_sizes[p]*passenger_position_trains[i][t][p] for p in passengers) <= train_capacities[t]

        # Constraint: Station capacities
        for i in range(max_rounds):
            for s in stations:
                m += xsum(train_position_stations[i][s][t] for t in trains) <= station_capacities[s]

        # Constraint: Line capacities
        for i in range(max_rounds):
            for l in lines:
                m += xsum(train_position_lines[i][l][t] for t in trains) <= line_capacities[l]

        # Constraint: Passengers cannot change between two different stations
        for i in range(max_rounds-1):
            for p in passengers:
                for s1 in stations:
                    for s2 in stations:
                        if s1 == s2:
                            continue
                        m += passenger_position_stations[i][s1][p] <= (1-passenger_position_stations[i+1][s2][p])

        # Constraint: Passengers cannot change between two different trains
        for i in range(max_rounds-1):
            for p in passengers:
                for t1 in trains:
                    for t2 in trains:
                        if t1 == t2:
                            continue
                        m += passenger_position_trains[i][t1][p] <= (1-passenger_position_trains[i+1][t2][p])

        # Constraint: Passengers do not move after destination reached
        for p in passengers:
            for i in range(max_rounds-1):
                m += passenger_position_stations[i+1][targets[p]][p] - passenger_position_stations[i][targets[p]][p] >= 0

        # Constraint: All trains are at one position at a time
        for t in trains:
            for i in range(max_rounds):
                m += xsum(train_position_stations[i][s][t] for s in stations) + xsum(train_position_lines[i][l][t] for l in lines) == 1

        # Constraint: All passengers are at one position at a time
        for p in passengers:
            for i in range(max_rounds):
                m += xsum(passenger_position_stations[i][s][p] for s in stations) + xsum(passenger_position_trains[i][t][p] for t in trains) == 1

        # Constraint: All trains are at their initial positions in the first round. If no position specified, use any station
        for t in trains:
            if train_initial_positions[t] is not None:
                m += train_position_stations[0][train_initial_positions[t]][t] == 1
            else:
                m += xsum(train_position_stations[0][s][t] for s in stations) == 1

        # Constraint: All passenger groups are at their initial positions
        for p in passengers:
            m += passenger_position_stations[0][passenger_initial_positions[p]][p] == 1

        # Constraint: All passenger groups are at their final positions
        for p in passengers:
            m += passenger_position_stations[-1][targets[p]][p] == 1

        # Constraint: Time is updated if destination not reached
        for p in passengers:
            for i in range(max_rounds-1):
                m += passenger_time[i+1][p] == (passenger_time[i][p] + 1 - passenger_position_stations[i][targets[p]][p])

        # Constraint: Calculate passenger delay
        m.objective = minimize(xsum(
            passenger_delay[p] for p in passengers
        ))

        print(f'model has {m.num_cols} vars, {m.num_rows} constraints and {m.num_nz} nzs')
        status = m.optimize()
        if status == OptimizationStatus.OPTIMAL:
            print(f'optimal solution cost {m.objective_value} found')
        elif status == OptimizationStatus.FEASIBLE:
            print(f'sol.cost {m.objective_value} found, best possible: {m.objective_bound}')
        elif status == OptimizationStatus.INFEASIBLE:
            raise Exception("model infeasable")
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            raise Exception("no solution found")


        #for t in trains:
        #    for i in range(max_rounds):
        #        for v in stations:
        #            if train_position_stations[i][v][t].x == 1:
        #                print(f"{i}: T{t+1} S{v+1}")
        #        for l in lines:
        #            if train_position_lines[i][l][t].x == 1:
        #                print(f"{i}: T{t+1} L{l+1} prog {train_progress[i][l][t].x}")
        #print()
        #for p in passengers:
        #    for i in range(max_rounds):
        #        for v in stations:
        #            if passenger_position_stations[i][v][p].x == 1:
        #                print(f"{i}: P{p+1} S{v+1}")
        #        for t in trains:
        #            if passenger_position_trains[i][t][p].x == 1:
        #                print(f"{i}: P{p+1} T{t+1}")

        actions = defaultdict(RoundAction)
        current_state = deepcopy(network_state)
        for i in range(max_rounds):
            action = RoundAction()

            if i == 0:
                for t in trains:
                    if network_state.trains[aid(t, 'T')].position_type == TrainPositionType.NOT_STARTED:
                        for s in stations:
                            if train_position_stations[i][s][t].x == 1:
                                action.train_starts[aid(t, 'T')] = aid(s, 'S')
            else:
                for t in trains:
                    current_train = current_state.trains[aid(t, 'T')]
                    for s in stations:
                        if train_position_stations[i][s][t].x == 1:
                            position_type = TrainPositionType.STATION
                            position = aid(s, 'S')
                    for l in lines:
                        if train_position_lines[i][l][t].x == 1:
                            position_type = TrainPositionType.LINE
                            position = aid(l, 'L')
                    if current_train.position_type == TrainPositionType.STATION and current_train.position != position:
                        if position_type == TrainPositionType.LINE:
                            action.train_departs[current_train.name] = position
                        elif position_type == TrainPositionType.STATION:
                            action.train_departs[current_train.name] = network_graph[current_train.position][position]["name"]

                for p in passengers:
                    current_passenger = current_state.passenger_groups[aid(p, 'P')]
                    for s in stations:
                        if passenger_position_stations[i][s][p].x == 1:
                            position_type = PassengerGroupPositionType.STATION
                            position = aid(s, 'S')
                    for t in trains:
                        if passenger_position_trains[i][t][p].x == 1:
                            position_type = PassengerGroupPositionType.TRAIN
                            position = aid(t, 'T')
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
