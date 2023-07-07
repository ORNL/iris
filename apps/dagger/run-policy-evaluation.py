#!/usr/bin/env python3

"""Wrapper script to run DAGGER at a high level. Similar to run-policy-evaluation."""
__author__ = "Aaron Young"
__copyright__ = "Copyright (c) 2020-2023, Oak Ridge National Laboratory (ORNL) Architectures and Performance Group (APG). All rights reserved."
__license__ = "GPL"
__version__ = "1.0"

import argparse
import sys
import os
import datetime
import subprocess
import time
import functools
import random
import iris
import re
import traceback
from multiprocessing import Pool
from subprocess import check_output

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

_payloads = {
    "linear_10": {
        "name": "Linear 10",
        "file": "linear10-graph",
        "generator_args": '--kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth=10 --num-tasks=10 --min-width=1 --max-width=1',
        "runner_args": '--graph="dagger-payloads/linear10-graph.json" --logfile="time.csv" --repeats=1 --size=256  --kernels="ijk" --duplicates="0" --buffers-per-kernel="ijk:w r r" --kernel-dimensions="ijk:2" --kernel-split="100" --depth=10 --num-tasks=10 --min-width=1 --max-width=1',
    }
}

def run(command, verbose=True, noop=False):
    """Print command then run command"""
    return_val = ''

    if verbose:
        print(command)
    if not noop:
        try:
            return_val = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE).decode()
        except subprocess.CalledProcessError as e:
            err_mesg = f'{os.getcwd()}: {e}\n\n{traceback.format_exc()}\n\n{e.returncode}\n\n{e.stdout.decode()}\n\n{e.stderr.decode()}'
            print(err_mesg, file=sys.stderr)
            with open('err.txt', 'w') as fd:
                fd.write(err_mesg)
            raise e
        except Exception as e:
            err_mesg = f'{os.getcwd()}: {e}\n\n{traceback.format_exc()}'
            print(err_mesg, file=sys.stderr)
            with open('err.txt', 'w') as fd:
                fd.write(err_mesg)
            raise e
        if verbose and return_val:
            print(return_val)

    return return_val

def init_parser(parser):
    parser.add_argument('payloads', metavar='p', type=str, nargs='*', help='Graphs to run. "all" runs everything.')
    parser.add_argument('--payload-dir', type=str, default=os.path.join(SCRIPT_DIR,"dagger-payloads"), help='Payloads directory')
    parser.add_argument('--results-dir', type=str, default=os.path.join(SCRIPT_DIR,"dagger-results"), help='Results directory')
    parser.add_argument('--graphs-dir', type=str, default=os.path.join(SCRIPT_DIR,"dagger-graphs"), help='Results directory')
    parser.add_argument('-f', '--force', action='store_true', help='Force regeneration of payloads.')
    parser.add_argument('-l', '--list', action='store_true', help='List all the graphs availible to run.')
    parser.add_argument('--policy', default='roundrobin', choices=["roundrobin", "depend", "profile", "random", "any", "all"], help='Scheduling policy to use')
    parser.add_argument("--attach-debugger",action='store_true',help="Attach debugger on port 5678.")

def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Run dagger generator and dagger runner at a higher level of abstraction.""")
    init_parser(parser)
    args = parser.parse_args(argv[1:])

    if args.attach_debugger:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        # debugpy.breakpoint()

    all_payloads = sorted(_payloads.keys())

    if args.list:
        print("All availible payloads:")
        print(all_payloads)
        exit(0)

    if 'all' in args.payloads or len(args.payloads) == 0:
        args.payloads = all_payloads

    run(f"mkdir -p {args.results_dir} {args.graphs_dir} {args.payload_dir}")
    print(f"Running DAGGER evaluation.... (graph figures can be found in {args.graphs_dir})")
    print(f"Payloads: {args.payloads}")

    #Only run DAGGER once to generate the payloads to test the systems (we want to compare the scheduling algorithms over different systems, and so we should fix the payloads over the whole experiment)
    #remove the dagger-payloads directory to regenerate payloads
    # Generate payloads.
    if not os.path.exists(args.payload_dir) or args.force:
        print(f"Generating DAGGER payloads in {args.payload_dir} (delete this directory to regenerate new DAG payloads)...")
        for payload in args.payloads:
            graph_path = os.path.join(args.payload_dir, _payloads[payload]['file']+'.json')
            run(f'{SCRIPT_DIR}/dagger_generator.py {_payloads[payload]["generator_args"]} --graph={graph_path}')
            run(f'cat {graph_path}')
            run(f'cp dag.png {os.path.join(args.graphs_dir, _payloads[payload]["file"]+".png")}')

    # Run payloads.
    for payload in args.payloads:
        print(f"Running IRIS on {payload} with Policy: {args.policy}.")
        graph_path = os.path.join(args.payload_dir, _payloads[payload]['file']+'.json')
        results_path = f'{args.results_dir}/{_payloads[payload]["file"]}-{args.policy}-{os.getenv("SYSTEM")}.csv'
        os.environ["IRIS_HISTORY"] = '1'

        run(f'{SCRIPT_DIR}/dagger_runner.py {_payloads[payload]["runner_args"]} --graph={graph_path} --scheduling-policy={args.policy}')
        # from dagger_runner import run as run_dagger
        # from dagger_runner import create_graph
        # from dagger_runner import parse_args as run_dagger_parse_args
        # os.environ["IRIS_KERNEL_DIR"] = SCRIPT_DIR
        # rargs = run_dagger_parse_args(f'{_payloads[payload]["runner_args"]} --graph={graph_path} --scheduling-policy={args.policy}')
        # # run_dagger(rargs)
        # iris.init()
        # for t in range(rargs.repeats):
        #     graph, input = create_graph(rargs)
        #     graph.submit()
        #     graph.wait()
        # print("Success")
        # iris.finalize()

        run(f'mv app-{os.getenv("SYSTEM")}*.csv {results_path}')
        #plot timeline with gantt
        run(f'python {SCRIPT_DIR}/gantt/gantt.py --dag={graph_path} --timeline={results_path} --combined-out={args.graphs_dir}/{_payloads[payload]["file"]}-{args.policy}-{os.getenv("SYSTEM")}.pdf')

if __name__ == '__main__':
    main(sys.argv)
