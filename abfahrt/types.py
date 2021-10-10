from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrainPositionType(Enum):
    STATION = 0
    LINE = 1
    NOT_STARTED = 2

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class PassengerGroupPositionType(Enum):
    STATION = 0
    TRAIN = 1
    DESTINATION_REACHED = 2

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass(order=True, unsafe_hash=True)
class Train:
    name: str
    position_type: TrainPositionType
    speed: float
    capacity: int
    position: Optional[str]
    next_station: Optional[str]
    line_progress: float = 0
    passenger_groups: list[str] = field(default_factory=list, compare=False)

    def is_valid(self, game_state: 'NetworkState') -> bool:
        if self.position_type not in [
            TrainPositionType.STATION, TrainPositionType.LINE, TrainPositionType.NOT_STARTED]:
            return False
        if self.position_type == TrainPositionType.STATION and self.position not in game_state.stations:
            return False
        if self.position_type == TrainPositionType.LINE and (
            self.position not in game_state.lines or self.line_progress is None or self.line_progress < 0 or self.line_progress >=
            game_state.lines[self.position].length or self.next_station is None):
            return False
        if self.capacity < 0:
            return False
        if self.speed <= 0:
            return False
        if not isinstance(self.passenger_groups, list):
            return False
        if any(
            [group not in game_state.passenger_groups for group in self.passenger_groups]):
            return False
        if sum([game_state.passenger_groups[group].group_size for group in self.passenger_groups]
               ) > self.capacity:
            return False
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "position": self.position if self.position is not None else "",
                "capacity": self.capacity, "speed": self.speed,
                "next_station": self.next_station if self.next_station is not None else "",
                "passenger_groups": [passenger_group for passenger_group in self.passenger_groups]}


@dataclass(order=True, unsafe_hash=True)
class Station:
    name: str
    capacity: int
    trains: list[str] = field(default_factory=list, compare=False)
    passenger_groups: list[str] = field(default_factory=list, compare=False)

    def is_full(self) -> bool:
        return len(self.trains) == self.capacity

    def get_free_capacity(self) -> int:
        return self.capacity - len(self.trains)

    def is_valid(self, game_state: 'NetworkState') -> bool:
        if self.capacity < 0:
            return False
        if not isinstance(self.trains, list):
            return False
        if len(self.trains) > self.capacity:
            return False
        if any([train not in game_state.trains for train in self.trains]):
            return False
        if not isinstance(self.passenger_groups, list):
            return False
        if any(
            [group not in game_state.passenger_groups for group in self.passenger_groups]):
            return False
        return True

    def to_dict(self) -> dict:
        return {'name': self.name, "capacity": self.capacity,
                "trains": [train for train in self.trains],
                "passenger_groups": [passenger_group for passenger_group in self.passenger_groups]}


@dataclass(order=True)
class PassengerGroup:
    name: str
    position: str
    position_type: PassengerGroupPositionType
    group_size: int
    destination: str
    time_remaining: int

    def is_valid(self, game_state: 'NetworkState') -> bool:
        if self.position_type not in [
            PassengerGroupPositionType.STATION, PassengerGroupPositionType.TRAIN,
            PassengerGroupPositionType.DESTINATION_REACHED]:
            return False
        if self.position_type == PassengerGroupPositionType.STATION and self.position not in game_state.stations:
            return False
        if self.position_type == PassengerGroupPositionType.TRAIN and self.position not in game_state.trains:
            return False
        if self.position_type == PassengerGroupPositionType.DESTINATION_REACHED and self.position is not None:
            return False
        if self.group_size <= 0:
            return False
        if self.destination not in game_state.stations:
            return False
        return True

    def delay(self) -> int:
        if self.time_remaining >= 0:
            return 0
        return -self.time_remaining * self.group_size

    def is_destination_reached(self) -> bool:
        return self.position_type == PassengerGroupPositionType.DESTINATION_REACHED

    def to_dict(self) -> dict:
        return {"name": self.name, "position": self.position,
                "group_size": self.group_size, "destination": self.destination,
                "time_remaining": self.time_remaining}


@dataclass(unsafe_hash=True)
class Line:
    name: str
    length: float
    start: str  # station id
    end: str  # station id
    capacity: int
    trains: list[str] = field(default_factory=list, compare=False)

    def is_full(self) -> bool:
        return len(self.trains) == self.capacity

    def get_free_capacity(self) -> int:
        return self.capacity - len(self.trains)

    def is_valid(self, game_state: 'NetworkState') -> bool:
        if self.length <= 0:
            return False
        if self.start not in game_state.stations or self.end not in game_state.stations:
            return False
        if self.capacity < 0:
            return False
        if not isinstance(self.trains, list):
            return False
        if any([train not in game_state.trains for train in self.trains]):
            return False
        if len(self.trains) > self.capacity:
            return False
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "length": self.length, "start": self.start, "end": self.name,
                "capacity": self.capacity, "trains": [train for train in self.trains]}


@dataclass()
class NetworkState:
    trains: dict[str, Train] = field(default_factory=dict)
    passenger_groups: dict[str, PassengerGroup] = field(default_factory=dict)
    stations: dict[str, Station] = field(default_factory=dict)
    lines: dict[str, Line] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return all([train.is_valid(self) for train in self.trains.values()]) and all(
            [station.is_valid(self) for station in self.stations.values()]) and all(
            [line.is_valid(self) for line in self.lines.values()]) and all(
            [passenger_group.is_valid(self) for passenger_group in self.passenger_groups.values()])

    def is_finished(self):
        return all([passenger_group.is_destination_reached()
                    for passenger_group in self.passenger_groups.values()])

    def apply_all(self, schedule: 'Schedule'):
        num_rounds = max(schedule.actions.keys())
        start_round = 0 if schedule.actions[0].is_zero_round() else 1
        for i in range(start_round, num_rounds + 1):
            self.apply(schedule.actions[i])
            if not self.is_valid():
                raise Exception("invalid state after round", i)

    def apply(self, action: 'RoundAction'):
        if action.is_zero_round():
            for train_id in action.train_starts:
                train = self.trains[train_id]
                station = self.stations[action.train_starts[train_id]]
                if train.position_type != TrainPositionType.NOT_STARTED or train.position is not None:
                    raise Exception("cannot start train that has been started")
                train.position = station.name
                train.position_type = TrainPositionType.STATION
                station.trains.append(train.name)
            return

        for passenger_group in self.passenger_groups.values():
            if not passenger_group.is_destination_reached():
                passenger_group.time_remaining -= 1

        for train_id in action.train_departs:
            train = self.trains[train_id]
            if train.position_type != TrainPositionType.STATION:
                raise Exception("Cannot depart train that is not in station")
            line = self.lines[action.train_departs[train_id]]
            station = self.stations[train.position]

            line.trains.append(train.name)
            station.trains.remove(train_id)
            train.next_station = line.start if line.start != train.position else line.end
            train.position = line.name
            train.position_type = TrainPositionType.LINE
        for group_id in action.passenger_detrains:
            passenger_group = self.passenger_groups[group_id]
            train = self.trains[passenger_group.position]
            if passenger_group.position_type != PassengerGroupPositionType.TRAIN or train.position_type != TrainPositionType.STATION:
                raise Exception(
                    "passenger group must be in a train that is in a station to detrain")
            station = self.stations[train.position]

            train.passenger_groups.remove(passenger_group.name)
            if station.name == passenger_group.destination:
                passenger_group.position = None
                passenger_group.position_type = PassengerGroupPositionType.DESTINATION_REACHED
            else:
                passenger_group.position_type = PassengerGroupPositionType.STATION
                passenger_group.position = train.position
                station.passenger_groups.append(group_id)
        for passenger_group_id in action.passenger_boards:
            passenger_group = self.passenger_groups[passenger_group_id]
            train = self.trains[action.passenger_boards[passenger_group_id]]
            if train.position_type != TrainPositionType.STATION or passenger_group.position_type != PassengerGroupPositionType.STATION or passenger_group.position != train.position:
                raise Exception(
                    "passenger group and train must be at the same station to board")
            station = self.stations[passenger_group.position]

            passenger_group.position_type = PassengerGroupPositionType.TRAIN
            passenger_group.position = train.name
            station.passenger_groups.remove(passenger_group_id)
            train.passenger_groups.append(passenger_group_id)

        for train in self.trains.values():
            if train.position_type == TrainPositionType.LINE:
                if train.line_progress + train.speed >= self.lines[train.position].length:
                    train.position_type = TrainPositionType.STATION
                    self.lines[train.position].trains.remove(train.name)
                    train.position = train.next_station
                    self.stations[train.position].trains.append(train.name)
                    train.next_station = None
                    train.line_progress = 0
                else:
                    train.line_progress += train.speed

    def to_dict(self) -> dict:
        return {"trains": {name: train.to_dict()
                           for (name, train) in self.trains.items()},
                "lines": {name: line.to_dict() for (name, line) in self.lines.items()},
                "passenger_groups": {name: passenger_group.to_dict() for (name, passenger_group) in
                                     self.passenger_groups.items()},
                "stations": {name: station.to_dict() for (name, station) in self.stations.items()}}

    def serialize(self) -> str:
        # Note: this only works properly on initial game states (such as ones
        # generated randomly)
        output = ""
        output += "[Stations]\n"
        for station in self.stations.values():
            output += f"{station.name} {station.capacity}\n"
        output += "\n[Lines]\n"
        for line in self.lines.values():
            output += f"{line.name} {line.start} {line.end} {line.length} {line.capacity}\n"
        output += "\n[Trains]\n"
        for train in self.trains.values():
            output += f"{train.name} {train.position if train.position_type == TrainPositionType.STATION else '*'} {train.speed} {train.capacity}\n"
        output += "\n[Passengers]\n"
        for passenger in self.passenger_groups.values():
            output += f"{passenger.name} {passenger.position} {passenger.destination} {passenger.group_size} {passenger.time_remaining}\n"
        return output

    def total_delay(self):
        return sum(
            [group.delay() for group in self.passenger_groups.values()])


@dataclass()
class RoundAction:
    train_starts: dict[str, str] = field(default_factory=dict)
    train_departs: dict[str, str] = field(default_factory=dict)
    passenger_boards: dict[str, str] = field(default_factory=dict)
    passenger_detrains: list[str] = field(default_factory=list)

    def is_zero_round(self):
        is_zero_round = len(self.train_starts) != 0
        if is_zero_round and (len(self.train_departs) != 0 or len(
            self.passenger_boards) != 0 or len(self.passenger_detrains) != 0):
            raise Exception(
                "invalid round: should be zero round, but more actions")
        return is_zero_round

    def serialize(self, current_round: int) -> [dict, dict]:
        train_actions, passenger_actions = dict(), dict()
        for train_id, station_id in self.train_starts.items():
            train_actions[train_id] = f"{current_round} Start {station_id}"
        for train_id, line_id in self.train_departs.items():
            train_actions[train_id] = f"{current_round} Depart {line_id}"
        for passenger_id, train_id in self.passenger_boards.items():
            passenger_actions[passenger_id] = f"{current_round} Board {train_id}"
        for passenger_id in self.passenger_detrains:
            passenger_actions[passenger_id] = f"{current_round} Detrain"
        return train_actions, passenger_actions


@dataclass
class Schedule:
    actions: defaultdict[int, RoundAction]

    @classmethod
    def from_dict(cls, dictionary):
        return Schedule(defaultdict(RoundAction, dictionary))

    def serialize(self) -> str:
        all_train_actions, all_passenger_actions = dict(), dict()

        num_rounds = max(self.actions.keys())
        start_round = 0 if self.actions[0].is_zero_round() else 1
        for i in range(start_round, num_rounds + 1):
            current_round_action = self.actions[i]
            train_actions, passenger_actions = current_round_action.serialize(i)
            for train_id, train_action in train_actions.items():
                if train_id in all_train_actions:
                    all_train_actions[train_id].append(train_action)
                else:
                    all_train_actions[train_id] = [train_action]

            for passenger_id, passenger_action in passenger_actions.items():
                if passenger_id in all_passenger_actions:
                    all_passenger_actions[passenger_id].append(passenger_action)
                else:
                    all_passenger_actions[passenger_id] = [passenger_action]

        output = ""
        for train_id, all_actions in all_train_actions.items():
            output += f"[Train:{train_id}]\n"
            output += '\n'.join(all_actions)
            output += '\n\n'

        for passenger_id, all_actions in all_passenger_actions.items():
            output += f"[Passenger:{passenger_id}]\n"
            output += '\n'.join(all_actions)
            output += '\n\n'

        return output
