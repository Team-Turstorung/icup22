from enum import Enum
from typing import Union, Optional

import networkx as nx


class TrainPositionType(Enum):
    STATION = 0
    LINE = 1
    NOT_STARTED = 2


class PassengerGroupPositionType(Enum):
    STATION = 0
    TRAIN = 1


class Train:
    def __init__(self, name: str, position: Union['Station', 'Line', None], position_type: TrainPositionType,
                 speed: float, capacity: int,
                 passenger_groups: list['PassengerGroup'], line_progress: Optional[int] = None,
                 next_station: Optional['Station'] = None):

        self.name = name
        self.position = position
        self.position_type = position_type
        self.next_station = next_station  # makes sense when on line
        self.line_progress = line_progress  # distance driven on line

        self.speed = speed
        self.capacity = capacity
        if passenger_groups is None:
            self.passenger_groups = []
        else:
            self.passenger_groups = passenger_groups

    def is_valid(self) -> bool:
        if self.position_type not in [TrainPositionType.STATION, TrainPositionType.LINE, TrainPositionType.NOT_STARTED]:
            return False
        if self.position_type == TrainPositionType.STATION and (
            type(self.position).__name__ != 'Station' or self.position is None or self.line_progress is not None):
            return False
        if self.position_type == TrainPositionType.LINE and (
            type(
                self.position).__name__ != 'Line' or self.position is None or self.line_progress is None or self.line_progress < 0 or self.line_progress >= self.position.length):
            return False
        if self.next_station is None:
            return False
        if self.capacity < 0:
            return False
        if self.speed <= 0:
            return False
        if type(self.passenger_groups) != list:
            return False
        if any([type(group) != PassengerGroup for group in self.passenger_groups]):
            return False
        if sum([group.group_size for group in self.passenger_groups]) > self.capacity:
            return False
        return True


class Station:
    def __init__(self, name: str, capacity: int, trains: list[Train], passenger_groups: list['PassengerGroup']):
        self.name = name
        self.capacity = capacity
        self.trains = trains
        self.passenger_groups = passenger_groups

    def is_valid(self) -> bool:
        if self.capacity < 0:
            return False
        if type(self.trains) != list:
            return False
        if len(self.trains) > self.capacity:
            return False
        if any([type(train) != Train for train in self.trains]):
            return False
        if any([type(group) != Train for group in self.passenger_groups]):
            return False
        return True


class PassengerGroup:
    def __init__(self, name: str, position: Union[Train, Station], position_type: PassengerGroupPositionType,
                 group_size: int, destination: Station, time_remaining: int):
        self.name = name
        self.position = position  # train or station passenger is on
        self.position_type = position_type
        self.group_size = group_size
        self.destination = destination
        self.time_remaining = time_remaining

    def is_valid(self) -> bool:
        if self.position_type not in [PassengerGroupPositionType.STATION, PassengerGroupPositionType.TRAIN]:
            return False
        if self.position_type == PassengerGroupPositionType.STATION and type(self.position) != Station:
            return False
        if self.position_type == PassengerGroupPositionType.TRAIN and type(self.position) != Train:
            return False
        return True

    def is_destination_reached(self) -> bool:
        return self.position == self.destination

    def __str__(self):
        return 'PassengerGroup{' + self.name + '}'

    def __repr__(self):
        return self.__str__()


class Line:
    def __init__(self, name: str, length: float, start: Station, end: Station, capacity: int, trains: list[Train]):
        self.name = name
        self.length = length
        self.start = start
        self.end = end
        self.capacity = capacity
        self.trains = trains

    def is_valid(self):
        if self.length <= 0:
            return False
        if type(self.start) != Station or type(self.end) != Station:
            return False
        if self.capacity < 0:
            return False
        if any([type(train) != Train for train in self.trains]):
            return False
        if len(self.trains) > self.capacity:
            return False
        return True


class GameState:
    def __init__(self, graph: nx.Graph, trains: dict[str, Train], passenger_groups: dict[str, PassengerGroup],
                 stations: dict[str, Station], lines: dict[str, Line]):
        self.graph = graph  # stations and lines and how they are connected (doesn't change)
        self.trains = trains
        self.passenger_groups = passenger_groups
        self.stations = stations
        self.lines = lines

    def is_valid(self):
        return all([train.is_valid() for train in self.trains.values()]) and all(
            [station.is_valid() for station in self.stations.values()]) and all(
            [line.is_valid for line in self.lines.values()]) and all(
            [passenger_group.is_valid() for passenger_group in self.passenger_groups.values()])

    def is_finished(self):
        return all([passenger_group.is_destination_reached() for passenger_group in self.passenger_groups.values()])

    def __str__(self):
        return 'GameState={' + ', '.join(
            [str(self.stations), str(self.lines), str(self.trains), str(self.passenger_groups)]) + '}'


class RoundAction:
    def __init__(self, train_starts: dict[str, str], train_departs: dict[str, str], passenger_boards: dict[str, str],
                 passenger_detrains: list[str]):
        self.train_starts = train_starts  # only from * station
        self.train_departs = train_departs
        self.passenger_boards = passenger_boards
        self.passenger_detrains = passenger_detrains


class Schedule:
    def __init__(self, round_actions: list[RoundAction]):
        self.round_actions = round_actions
