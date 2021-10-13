from mip import Model, xsum, minimize, BINARY

import networkx as nx
from networkx.algorithms import single_source_dijkstra_path_length, bidirectional_dijkstra
from networkx.algorithms.assortativity.correlation import numeric_assortativity_coefficient

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction

from dataclasses import asdict

class MipSolver(Solution):
    def schedule(self, network_state: NetworkState, graph: nx.graph):
        max_rounds = 10
        # constants
        stations = [ int(station_id[1:])-1 for station_id in network_state.stations.keys() ]
        trains = [ int(train_id[1:])-1 for train_id in network_state.trains.keys() ]
        passengers = [ int(passenger_id[1:])-1 for passenger_id in network_state.passenger_groups.keys() ]
        target_times = { int(passenger_id[1:])-1: passenger.time_remaining for passenger_id, passenger in network_state.passenger_groups.items() }
        targets = { int(passenger_id[1:])-1: int(passenger.destination[1:])-1 for passenger_id, passenger in network_state.passenger_groups.items() }
       
        adjacency_matrix = nx.adjacency_matrix(graph, weight=None)
        print(adjacency_matrix[1, 0])   
        
        
        m = Model()
        
        train_position = [[[m.add_var(var_type=BINARY) for t in trains]for s in stations] for k in range(max_rounds)]
        passenger_position_stations = [[[m.add_var(var_type= BINARY) for p in passengers]for s in stations] for k in range(max_rounds)]
        passenger_position_trains = [[[m.add_var(var_type=BINARY) for p in passengers] for t in trains] for k in range(max_rounds)]
        passenger_time = [[m.add_var() for p in passengers] for k in range(max_rounds)]

        for i in range(max_rounds-1):
            for s1 in stations:             
                for s2 in stations:
                    for t in trains:
                        m += train_position[i][s1][t] + train_position[i+1][s2][t] <= 1+adjacency_matrix[s1, s2]
                    
                    for p in passengers:
                        has_corresponding_train = passenger_position_stations[i][s1][p]+train_position[i][s1][0]+passenger_position_stations[i+1][s2][p]+train_position[i+1][s2][0] == 4
                        for t in trains:
                            has_corresponding_train = has_corresponding_train or passenger_position_stations[i][s1][p]+train_position[i][s1][t]+passenger_position_stations[i+1][s2][p]+train_position[i+1][s2][t] == 4
                        if s1 == s2:
                            m += passenger_position_stations[i][s1][p] == passenger_position_stations[i][s2][p] or has_corresponding_train

       
        for t in trains:
            for i in range(max_rounds):
                m += xsum(train_position[i][s][t] for s in stations) == 1
       
        for p in passengers:
            for i in range(max_rounds):
                m += xsum(passenger_position_stations[i][s][p] for s in stations) == 1

        for p in passengers:
            for i in range(max_rounds-1):
                m += xsum(passenger_position_stations[i][s][p] * targets[p] for s in stations) == 0 or passenger_time[i][p] == passenger_time[i+1][p]

                m += xsum(passenger_position_stations[i][s][p] * targets[p] for s in stations) != 0 or passenger_time[i][p]+1 == passenger_time[i+1][p]


        m.objective = minimize(xsum(
            passenger_time[max_rounds-1][p]-target_time for p, target_time in target_times.items()
        ))

        m.optimize()