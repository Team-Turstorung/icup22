from enum import Enum
from typing import Union
from dataclasses import dataclass, field


class TrainPositionType(Enum):
    STATION = 0
    LINE = 1
    NOT_STARTED = 2


class PassengerGroupPositionType(Enum):
    STATION = 0
    TRAIN = 1


@dataclass
class Train:
    name: str
    position_type: TrainPositionType
    speed: float
    capacity: int
    position: Union['Station', 'Train', None]
    line_progress: float = 0
    next_station: 'Station' = None
    passenger_groups: list['PassengerGroup'] = field(default_factory=list)

    def is_valid(self) -> bool:
        if self.position_type not in [
                TrainPositionType.STATION, TrainPositionType.LINE, TrainPositionType.NOT_STARTED]:
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
        if not isinstance(self.passenger_groups, list):
            return False
        if any([not isinstance(group, PassengerGroup)
                for group in self.passenger_groups]):
            return False
        if sum([group.group_size for group in self.passenger_groups]
               ) > self.capacity:
            return False
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "position": self.position.name if self.position is not None else "",
                "capacity": self.capacity, "speed": self.speed, "next_station": self.next_station.name if self.next_station is not None else "", "passenger_groups": [passenger_group.name for passenger_group in self.passenger_groups]}


@dataclass()
class Station:
    name: str
    capacity: int
    trains: list['Train'] = field(default_factory=list)
    passenger_groups: list['PassengerGroup'] = field(default_factory=list)

    def is_valid(self) -> bool:
        if self.capacity < 0:
            return False
        if not isinstance(self.trains, list):
            return False
        if len(self.trains) > self.capacity:
            return False
        if any([not isinstance(train, Train) for train in self.trains]):
            return False
        if any([not isinstance(group, Train)
                for group in self.passenger_groups]):
            return False
        return True

    def to_dict(self) -> dict:
        return {'name': self.name, "capacity": self.capacity,
                "trains": [train.name for train in self.trains], "passenger_groups": [passenger_group.name for passenger_group in self.passenger_groups]}


@dataclass()
class PassengerGroup:
    name: str
    position: Union['Train', 'Station']
    position_type: PassengerGroupPositionType
    group_size: int
    destination: 'Station'
    time_remaining: int

    def is_valid(self) -> bool:
        if self.position_type not in [
                PassengerGroupPositionType.STATION, PassengerGroupPositionType.TRAIN]:
            return False
        if self.position_type == PassengerGroupPositionType.STATION and not isinstance(
                self.position, Station):
            return False
        if self.position_type == PassengerGroupPositionType.TRAIN and not isinstance(
                self.position, Train):
            return False
        return True

    def is_destination_reached(self) -> bool:
        return self.position == self.destination

    def __str__(self):
        return 'PassengerGroup{' + self.name + '}'

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {"name": self.name, "position": self.position.name,
                "group_size": self.group_size, "destination": self.destination.name, "time_remaining": self.time_remaining}


@dataclass()
class Line:
    name: str
    length: float
    start: 'Station'
    end: 'Station'
    capacity: int
    trains: list['Train'] = field(default_factory=list)

    def is_valid(self):
        if self.length <= 0:
            return False
        if not isinstance(self.start, Station) or not isinstance(
                self.end, Station):
            return False
        if self.capacity < 0:
            return False
        if any([not isinstance(train, Train) for train in self.trains]):
            return False
        if len(self.trains) > self.capacity:
            return False
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "length": self.length, "start": self.start.name, "end": self.end.name,
                "capacity": self.capacity, "trains": [train.name for train in self.trains]}


class GameState:
    def __init__(self, trains: dict[str, Train], passenger_groups: dict[str, PassengerGroup],
                 stations: dict[str, Station], lines: dict[str, Line]):
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
        return all([passenger_group.is_destination_reached()
                   for passenger_group in self.passenger_groups.values()])

    def apply(self, action: 'RoundAction'):
        if action.is_zero_round():
            for train, station in action.train_starts:
                self.trains[train].position = station
                self.trains[train].position_type = TrainPositionType.STATION
            return
        for train, next_station in action.train_departs:
            if self.trains[train].position_type != TrainPositionType.STATION:
                raise Exception("Cannot depart train that is not in station")
            current_position = self.trains[train].position
            next_position = self.trains[train].next_station
            for current_line in self.lines.values():
                line = current_line
                if current_line.start == current_position and current_line.end == next_position:
                    break
            self.trains[train].position = line
            self.trains[train].position_type = TrainPositionType.LINE
            self.trains[train].next_station = self.stations[next_station]
            line.trains.append(self.trains[train])
        # TODO: implement passenger_board, passenger_detrains

        for passenger_group in self.passenger_groups.values():
            passenger_group.time_remaining -= 1

    def __str__(self):
        return 'GameState={' + ', '.join(
            [str(self.stations), str(self.lines), str(self.trains), str(self.passenger_groups)]) + '}'

    def to_dict(self) -> dict:
        return {"trains": {name: train.to_dict()
                           for (name, train) in self.trains.items()}, "lines": {name: line.to_dict() for (name, line) in self.lines.items()}, "passenger_groups": {name: passenger_group.to_dict() for (name, passenger_group) in self.passenger_groups.items()}, "stations": {name: station.to_dict() for (name, station) in self.stations.items()}}

    def serialize(self) -> str:
        # Note: this only works properly on initial game states (such as ones
        # generated randomly)
        output = ""
        output += "[Stations]\n"
        for station in self.stations.values():
            output += f"{station.name} {station.capacity}\n"
        output += "\n[Lines]\n"
        for line in self.lines.values():
            output += f"{line.name} {line.start.name} {line.end.name} {line.length} {line.capacity}\n"
        output += "\n[Trains]\n"
        for train in self.trains.values():
            output += f"{train.name} {train.position.name if train.position_type == TrainPositionType.STATION else '*'} {train.speed} {train.capacity}\n"
        output += "\n[Passengers]\n"
        for passenger in self.passenger_groups.values():
            output += f"{passenger.name} {passenger.position.name} {passenger.destination.name} {passenger.group_size} {passenger.time_remaining}\n"
        return output


class RoundAction:
    def __init__(self, train_starts: dict[str, str], train_departs: dict[str, str], passenger_boards: dict[str, str],
                 passenger_detrains: list[str]):
        self.train_starts = train_starts  # only from * station
        self.train_departs = train_departs
        self.passenger_boards = passenger_boards
        self.passenger_detrains = passenger_detrains

    def is_zero_round(self):
        is_zero_round = len(self.train_starts) == 0
        if is_zero_round and (len(self.train_departs) != 0 or len(
                self.passenger_boards) != 0 or len(self.passenger_detrains) != 0):
            raise Exception(
                "invalid round: should be zero round, but more actions")
        return is_zero_round


class Schedule:
    def __init__(self, round_actions: list[RoundAction]):
        self.round_actions = round_actions
