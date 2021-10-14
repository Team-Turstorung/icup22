from mip import Model, xsum, minimize, BINARY, OptimizationStatus

import networkx as nx

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, TrainPositionType


def mid(abfahrt_id):
    return int(abfahrt_id[1:])-1

class MipSolver(Solution):
    def schedule(self, network_state: NetworkState, network_graph: nx.graph):
        max_rounds = 10
        # constants
        stations = [mid(station_id) for station_id in network_state.stations.keys()]
        trains = [mid(train_id) for train_id in network_state.trains.keys()]
        passengers = [mid(passenger_id) for passenger_id in network_state.passenger_groups.keys()]

        target_times = {mid(passenger_id): passenger.time_remaining for passenger_id, passenger in
                        network_state.passenger_groups.items()}
        group_sizes = {mid(passenger_id): passenger.group_size for passenger_id, passenger in
                        network_state.passenger_groups.items()}
        train_initial_positions = {mid(train_id): mid(train.position) if train.position_type == TrainPositionType.STATION else None for train_id, train in
                                   network_state.trains.items()}
        passenger_initial_positions = {mid(passenger_id): mid(passenger.position) for passenger_id, passenger in
                                       network_state.passenger_groups.items()}
        targets = {int(passenger_id[1:]) - 1: int(passenger.destination[1:]) - 1 for passenger_id, passenger in
                   network_state.passenger_groups.items()}

        adjacency_matrix = nx.adjacency_matrix(network_graph, weight=None)
        m = Model()

        train_position = [[[m.add_var(var_type=BINARY) for _ in trains]for _ in stations] for _ in range(max_rounds)]
        passenger_position_stations = [[[m.add_var(var_type=BINARY) for _ in passengers] for _ in stations] for _ in range(max_rounds)]
        passenger_position_trains = [[[m.add_var(var_type=BINARY) for _ in passengers] for _ in trains] for _ in range(max_rounds)]
        passenger_time = [[m.add_var() for _ in passengers] for _ in range(max_rounds)]
        passenger_delay = [m.add_var() for _ in passengers] # we use the implicit lower bound of 0 for the delay

        # Constraint: Set passenger delay to target time minus time elapsed
        for p in passengers:
            m += passenger_delay[p] == (passenger_time[max_rounds-1][p]-target_times[p])*group_sizes[p]

        # Constraint: Trains can stay where they are or travel along the lines.
        for i in range(max_rounds-1):
            for s1 in stations:
                for s2 in stations:
                    if s1 == s2:
                        continue
                    for t in trains:
                        m += train_position[i][s1][t] + train_position[i+1][s2][t] <= 1+adjacency_matrix[s1, s2]

        # Constraint: When a passenger hops onto a train, the train must be in the corresponding station in two turns. The same is true when getting out.
        for i in range(max_rounds-1):
            for p in passengers:
                for t in trains:
                    for s in stations:
                        m += passenger_position_stations[i][s][p] + passenger_position_trains[i + 1][t][p] <= 1 + \
                             train_position[i][s][t]
                        m += passenger_position_stations[i][s][p] + passenger_position_trains[i + 1][t][p] <= 1 + \
                             train_position[i + 1][s][t]
                        m += passenger_position_trains[i][t][p] + passenger_position_stations[i + 1][s][p] <= 1 + \
                             train_position[i][s][t]
                        m += passenger_position_trains[i][t][p] + passenger_position_stations[i + 1][s][p] <= 1 + \
                             train_position[i + 1][s][t]

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
                m += xsum(train_position[i][s][t] for s in stations) == 1

        # Constraint: All passengers are at one position at a time
        for p in passengers:
            for i in range(max_rounds):
                m += xsum(passenger_position_stations[i][s][p] for s in stations) + xsum(passenger_position_trains[i][t][p] for t in trains) == 1

        # Constraint: All trains are at their initial positions in the first round
        for t in trains:
            if train_initial_positions[t] is not None:
                m += train_position[0][train_initial_positions[t]][t] == 1

        # Constraint: All passenger groups are at their initial positions
        for p in passengers:
            m += passenger_position_stations[0][passenger_initial_positions[p]][p] == 1

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

        for i in range(max_rounds):
            for v in stations:
                for t in trains:
                    if train_position[i][v][t].x == 1:
                        print(f"{i}: T{t+1} S{v+1}")
        print()
        for p in passengers:
            for i in range(max_rounds):
                for v in stations:
                    if passenger_position_stations[i][v][p].x == 1:
                        print(f"{i}: P{p+1} S{v+1}")
                for t in trains:
                    if passenger_position_trains[i][t][p].x == 1:
                        print(f"{i}: P{p+1} T{t+1}")
