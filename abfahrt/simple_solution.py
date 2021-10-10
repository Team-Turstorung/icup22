from copy import deepcopy

import networkx as nx
from networkx.algorithms import single_source_dijkstra_path_length, bidirectional_dijkstra

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction
from tools.file_parser import parse_input_file


class SimpleSolver(Solution):
    def schedule(self, network_state: NetworkState,
                 network_graph: nx.Graph) -> Schedule:
        def get_closest_station_with_passengers(train_pos):
            sorted_dict = sorted(single_source_dijkstra_path_length(
                network_graph, train_pos).items(), key=lambda item: item[1])
            for station_id, _ in sorted_dict:
                if len(
                        network_state.stations[station_id].passenger_groups) > 0:
                    return station_id

        # step 0
        station_space_left = dict()
        for station_id, station in network_state.stations.items():
            if len(station.trains) < station.capacity:
                station_space_left[station_id] = station.capacity - \
                    len(station.trains)

        round_action = RoundAction()
        for train in network_state.trains.values():
            if train.position_type == TrainPositionType.NOT_STARTED:
                emptiest_station = max(
                    station_space_left.items(),
                    key=lambda item: item[1])[0]
                station_space_left[emptiest_station] -= 1
                round_action.train_starts[train.name] = emptiest_station

        network_state.apply(round_action)
        actions = dict()
        actions[0] = round_action

        # step 1
        max_capacity_train = max(
            network_state.trains.values(),
            key=lambda train: train.capacity)

        round_id = 0
        current_path = []
        while True:
            round_id += 1
            print(f'round id: {round_id}')
            round_action = RoundAction()

            if max_capacity_train.position_type == TrainPositionType.LINE:
                network_state.apply(round_action)
                actions[round_id] = round_action
                continue  # enjoying the ride

            if len(current_path) == 1:
                if len(max_capacity_train.passenger_groups) == 0:
                    # pick up
                    passenger_group = network_state.stations[max_capacity_train.position].passenger_groups[0]
                    round_action.passenger_boards[passenger_group] = max_capacity_train.name
                else:
                    # drop off
                    passenger_group = max_capacity_train.passenger_groups[0]
                    round_action.passenger_detrains.append(passenger_group)
                current_path = []
            elif len(current_path) == 0:
                # determine_next_target
                if len(max_capacity_train.passenger_groups) == 0:
                    # target closest group
                    station = get_closest_station_with_passengers(
                        max_capacity_train.position)
                    if station is None:
                        break  # all passengers are at destination, we are done!
                    _, current_path = bidirectional_dijkstra(
                        G=network_graph, source=max_capacity_train.position, target=station)
                else:
                    # target destination of passengers
                    station = network_state.passenger_groups[max_capacity_train.passenger_groups[0]].destination
                    _, current_path = bidirectional_dijkstra(
                        G=network_graph, source=max_capacity_train.position, target=station)
            else:
                # go to next station in path
                next_line = network_graph.edges[max_capacity_train.position,
                                                current_path[1]]['name']
                round_action.train_departs[max_capacity_train.name] = next_line
                current_path = current_path[1:]

            network_state.apply(round_action)
            actions[round_id] = round_action

        generated_schedule = Schedule(actions)
        return generated_schedule


if __name__ == '__main__':
    game_state, graph = parse_input_file(
        '/home/alex/informaticup2022/icup22/examples/official/simple/input.txt')
    solver = SimpleSolver()
    schedule = solver.schedule(deepcopy(game_state), graph)

    game_state.apply_all(schedule)
    print(f'final game state: {game_state}')