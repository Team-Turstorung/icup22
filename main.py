#!/usr/bin/env python3
import argparse
import logging
import os
import threading
from copy import deepcopy
from abfahrt.simple_solution import SimpleSolver
from abfahrt.simple_solution_multiple_trains import SimpleSolverMultipleTrains
from abfahrt.mip_solver import MipSolver
from abfahrt.simple_solution_random_multi import SimpleSolverMultipleTrainsRandom
from tools import generator, file_parser
from tools.file_parser import parse_input_file, parse_output_file
from tools.gui import start_gui

SOLUTIONS = {
    'simple': SimpleSolver,
    'multi': SimpleSolverMultipleTrains,
    'random': SimpleSolverMultipleTrainsRandom,
    'mip': MipSolver,
}


def solve(solver, input_file, output_file, simulate):
    if simulate and not output_file:
        raise Exception("cannot simulate without an output file")
    network_state, network_graph = file_parser.parse_input_file(input_file)
    solver = SOLUTIONS[solver](deepcopy(network_state), network_graph)
    sched = solver.schedule()
    network_state.apply_all(sched)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(sched.serialize())
    print("Schedule:")
    print(sched.serialize())
    print("Total delay using", type(solver).__name__, "is", network_state.total_delay())
    print()
    if simulate:
        print("Bahn-Simulator:")
        os.system(f"../informatiCup2022/Bahn-Simulator/Bahn-Simulator -input {input_file} -output {output_file}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    generator_parser = subparsers.add_parser('generate',
                                             help="""Generate a new scenario with the given parameters.
                                                     For a complete list of parameters take a look into the README""")
    generator.create_generator_parser(generator_parser)

    subparsers.add_parser('gui', help="Start the browser GUI")

    evaluate_parser = subparsers.add_parser('evaluate', help="Evaluate a solution given an input and an output file")
    evaluate_parser.add_argument('input', help="Input file for evaluation")
    evaluate_parser.add_argument('output', help="Output file for evaluation")

    solve_parser = subparsers.add_parser('solve', help="Solve a given input file with the selected solver")
    solve_parser.add_argument('solver', choices=SOLUTIONS.keys())
    solve_parser.add_argument('input')
    solve_parser.add_argument('output', nargs='?')
    solve_parser.add_argument('-g', '--gui', action='store_true')
    solve_parser.add_argument('-s', '--simulate', action='store_true')
    args = parser.parse_args()

    if args.command == 'generate':
        generator.execute(args)
    elif args.command == 'gui':
        start_gui()
    elif args.command == 'evaluate':
        state, graph = parse_input_file(args.input)
        schedule = parse_output_file(args.output)
        state.apply_all(schedule)
        print("Total delay:", state.total_delay())
    elif args.command == 'solve':
        if args.gui:
            threading.Thread(target=lambda: solve(args.solver, args.input, args.output, args.simulate)).start()
            start_gui()
        else:
            solve(args.solver, args.input, args.output, args.simulate)
