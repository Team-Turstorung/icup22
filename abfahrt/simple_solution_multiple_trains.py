import networkx as nx
from networkx.algorithms import all_pairs_dijkstra

from abfahrt.solution import Solution
from abfahrt.types import NetworkState, Schedule, TrainPositionType, RoundAction, Train, PassengerGroupPositionType, \
    PassengerGroup


class SimplesSolverMultipleTrains(Solution):
    def schedule(self, network_state: NetworkState, network_graph: nx.Graph) -> Schedule:

        def get_all_shortest_paths(network_graph: nx.Graph) -> dict[str, tuple]:
            shortest_paths = all_pairs_dijkstra(network_graph)
            all_shortest_paths = {}
            for path in shortest_paths:
                all_shortest_paths[path[0]] = path[1]
            return all_shortest_paths

        def place_wildcard_trains(network_state: NetworkState, shortest_paths: dict[str, tuple]) -> RoundAction:

            new_round_action = RoundAction()

            wildcard_trains = filter(lambda train: train.position_type == TrainPositionType.NOT_STARTED,
                                     network_state.trains.values())
            sorted_wildcard_trains = sorted(wildcard_trains, key=lambda train: train.speed, reverse=True)
            passenger_list = sorted(passenger_priorities.items(), reverse=True, key=lambda item: item[1])

            station_space_left = dict()
            for station_id, station in network_state.stations.items():
                station_space_left[station_id] = station.capacity - len(station.trains)

            for passenger_group_name, priority in passenger_list:
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

        def compute_priorities(passenger_groups: list[PassengerGroup]) -> dict[str, int]:
            priorities = dict()
            for passenger_group in passenger_groups:
                priorities[passenger_group.name] = all_shortest_paths[passenger_group.position][0][
                                                       passenger_group.destination] / (
                                                       passenger_group.time_remaining + 1) * passenger_group.group_size
            return priorities

        def navigate_train(train: Train, path: list[str]):
            if train.position_type == TrainPositionType.LINE or train.name not in on_tour:
                return
            if len(path) == 1:
                # if station we arrive at is full, depart some train to some line (there is always at least 1 free)
                if len(train.passenger_groups) == 0:
                    # pick up
                    board_passengers.append(train)
                else:
                    # drop off
                    detrain_passengers.append(train.passenger_groups[0])
                on_tour.remove(train.name)
            else:
                # go to next station in path
                next_line_id = network_graph.edges[train.position, path[1]]['name']
                round_action.train_departs[train.name] = next_line_id
                train_paths[train.name] = path[1:]

        def plan_train(train: Train):
            if len(train.passenger_groups) == 0:
                # go to suitable passenger group
                for passenger_group in sorted(network_state.waiting_passengers().values(),
                                              key=lambda passenger_group: passenger_priorities[passenger_group.name] / (
                                                  all_shortest_paths[train.position][0][passenger_group.position] + 1),
                                              reverse=True):
                    if passenger_group.name not in assigned_passenger_groups:
                        # go for it
                        return all_shortest_paths[train.position][1][passenger_group.position]
            else:
                # go to destination
                passenger_group = network_state.passenger_groups[train.passenger_groups[0]]
                return all_shortest_paths[train.position][1][passenger_group.destination]

        all_shortest_paths = get_all_shortest_paths(network_graph)
        passenger_priorities = compute_priorities(list(network_state.passenger_groups.values()))

        # all_relevant_nodes = set()
        # alternative_betweenness_centrality = {}
        # for passenger_group in network_state.passenger_groups.values():
        #     new_set = set(all_shortest_paths[passenger_group.position][1][passenger_group.destination])
        #     for node in new_set:
        #         if node not in alternative_betweenness_centrality:
        #             alternative_betweenness_centrality[node] = 1
        #         else:
        #             alternative_betweenness_centrality[node] += 1
        #     all_relevant_nodes = all_relevant_nodes.union(new_set)
        # print(sorted(alternative_betweenness_centrality.items(), key=lambda  item: item[1], reverse=True))
        # print(len(alternative_betweenness_centrality))
        # print(f"{max(alternative_betweenness_centrality.items(), key=lambda item: item[1])} is the maximum centrality. There are {len(network_state.passenger_groups)}")
        #
        # print(f"From {network_graph.number_of_nodes()} nodes There are {len(all_relevant_nodes)} nodes on shortest paths")

        # Create round action for zero Round
        round_action = place_wildcard_trains(network_state, all_shortest_paths)

        actions = dict()
        round_id = 0
        actions[round_id] = round_action
        network_state.apply(round_action)
        on_tour = set()
        train_paths = {}  # train: path
        board_passengers = []
        detrain_passengers = []
        while True:
            print(f"Processing round {round_id}")
            round_action = RoundAction()
            assigned_passenger_groups = set()  # mark waiting passenger groups that have a train coming for them
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

            for train in board_passengers:
                passenger_groups = network_state.stations[train.position].passenger_groups
                if not passenger_groups:
                    continue
                passenger_groups_sorted = sorted(passenger_groups, key=lambda name: passenger_priorities[name])
                for passenger_group in passenger_groups_sorted:
                    if network_state.passenger_groups[passenger_group].group_size <= train.capacity:
                        round_action.passenger_boards[passenger_group] = train.name
                        break

            board_passengers = []
            for passenger_group in detrain_passengers:
                round_action.passenger_detrains.append(passenger_group)
            detrain_passengers = []

            actions[round_id] = round_action
            network_state.apply(round_action)
            if not network_state.is_valid():
                raise Exception(f"invalid state at round {round_id}")
            if len(list(filter(lambda group: group.position_type != PassengerGroupPositionType.DESTINATION_REACHED,
                               network_state.passenger_groups.values()))) == 0:
                break
        schedule = Schedule.from_dict(actions)
        return schedule
