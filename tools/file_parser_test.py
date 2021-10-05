from unittest import TestCase

from tools.file_parser import parse_input_file, parse_output_file
from tools.game import TrainPositionType, PassengerGroupPositionType


def create_test_class(dataset):
    class TestClass(TestCase):
        def setUp(self):
            self.game_state, self.graph = parse_input_file(dataset[0])
            print(self.game_state.to_dict())

        def is_valid(self):
            self.assertTrue(self.game_state.is_valid())

        def test_stations(self):
            self.assertEqual(len(self.game_state.stations), len(dataset[1]))
            for data in dataset[1]:
                station = self.game_state.stations[data[0]]
                self.assertEqual(station.name, data[0])
                print(data[1])
                self.assertEqual(sorted(station.trains),
                                 data[1](self.game_state.trains))
                self.assertEqual(sorted(station.passenger_groups),
                                 data[2](self.game_state.passenger_groups))
                self.assertEqual(station.capacity, data[3])

        def test_lines(self):
            self.assertEqual(len(self.game_state.lines), len(dataset[2]))
            for data in dataset[2]:
                line = self.game_state.lines[data[0]]
                self.assertEqual(line.name, data[0])
                self.assertEqual(line.start, data[1](self.game_state.stations))
                self.assertEqual(line.end, data[2](self.game_state.stations))
                self.assertEqual(line.length, data[3])
                self.assertEqual(line.capacity, data[4])

        def test_trains(self):
            self.assertEqual(len(self.game_state.trains), len(dataset[3]))
            for data in dataset[3]:
                train = self.game_state.trains[data[0]]
                self.assertEqual(train.name, data[0])
                self.assertEqual(train.position_type, data[1])
                self.assertEqual(
                    train.position, data[2](
                        self.game_state.stations))
                self.assertEqual(train.speed, data[3])
                self.assertEqual(train.capacity, data[4])

        def test_passenger_groups(self):
            self.assertEqual(
                len(self.game_state.passenger_groups), len(dataset[4]))
            for data in dataset[4]:
                passenger_group = self.game_state.passenger_groups[data[0]]
                self.assertEqual(passenger_group.name, data[0])
                self.assertEqual(
                    passenger_group.position, data[1](
                        self.game_state.stations))
                self.assertEqual(
                    passenger_group.position_type,
                    PassengerGroupPositionType.STATION)
                self.assertEqual(
                    passenger_group.destination, data[2](
                        self.game_state.stations))
                self.assertEqual(passenger_group.group_size, data[3])
                self.assertEqual(passenger_group.time_remaining, data[4])

    return TestClass


TestSimple = create_test_class((
    'examples/official/simple/input.txt',
    [
        ('S1', lambda _: [], lambda _: [], 2),
        ('S2', lambda trains: [trains['T1']], lambda passenger_groups: [passenger_groups['P1'], passenger_groups['P2']],
         2),
        ('S3', lambda _: [], lambda _: [], 2)
    ],
    [
        ('L1', lambda stations: stations['S2'],
         lambda stations: stations['S3'], 3.14, 1),
        ('L2', lambda stations: stations['S2'],
         lambda stations: stations['S1'], 4, 1),
    ],
    [
        ('T1', TrainPositionType.STATION,
         lambda stations: stations['S2'], 5.5, 30),
        ('T2', TrainPositionType.NOT_STARTED,
         lambda stations: None, 0.9999999, 50),
    ],
    [
        ('P1', lambda stations: stations['S2'],
         lambda stations: stations['S3'], 3, 3),
        ('P2', lambda stations: stations['S2'],
         lambda stations: stations['S1'], 10, 3)
    ],
))

TestKapazitaet = create_test_class((
    'examples/official/kapazität/input.txt',
    [
        ('S1', lambda trains: sorted([trains['T1'], trains['T2'], trains['T3'], trains['T4'], trains['T5']]),
         lambda passenger_groups: sorted([passenger_groups['P1'], passenger_groups['P2'], passenger_groups['P3']]), 5),
        ('S2', lambda _: [], lambda _: [], 5),
    ],
    [
        ('L1', lambda stations: stations['S1'],
         lambda stations: stations['S2'], 1, 3),
    ],
    [
        ('T1', TrainPositionType.STATION,
         lambda stations: stations['S1'], 1, 1),
        ('T2', TrainPositionType.STATION,
         lambda stations: stations['S1'], 1, 1),
        ('T3', TrainPositionType.STATION,
         lambda stations: stations['S1'], 1, 1),
        ('T4', TrainPositionType.STATION,
         lambda stations: stations['S1'], 1, 1),
        ('T5', TrainPositionType.STATION,
         lambda stations: stations['S1'], 1, 1),
    ],
    [
        ('P1', lambda stations: stations['S1'],
         lambda stations: stations['S2'], 1, 4),
        ('P2', lambda stations: stations['S1'],
         lambda stations: stations['S2'], 1, 4),
        ('P3', lambda stations: stations['S1'],
         lambda stations: stations['S2'], 1, 4),
    ],
))


class TestSimpleOutput(TestCase):

    def setUp(self) -> None:
        self.schedule = parse_output_file('examples/official/simple/output.txt')

    def test_length(self):
        self.assertEqual(max(self.schedule.actions.keys()), 7)

    def test_train_departs(self):
        for i in range(8):
            self.assertDictEqual(self.schedule.actions[i].train_departs, {'T1': 'L1', 'T2': 'L2'} if i == 2 else {})

    def test_train_starts(self):
        for i in range(8):
            self.assertDictEqual(self.schedule.actions[i].train_starts, {'T2': 'S2'} if i == 0 else {})

    def test_passenger_detrains(self):
        for i in range(8):
            detrains = []
            if i == 3:
                detrains = ['P1']
            elif i == 7:
                detrains = ['P2']
            self.assertListEqual(self.schedule.actions[i].passenger_detrains, detrains)

    def test_passenger_boards(self):
        for i in range(8):
            self.assertDictEqual(self.schedule.actions[i].passenger_boards, {'P1': 'T1', 'P2': 'T2'} if i == 1 else {})
