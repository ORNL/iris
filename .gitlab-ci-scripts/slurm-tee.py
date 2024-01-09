#!/usr/bin/env python3

"""Print running slurm out until job completes."""

import argparse
import sys
import os
import datetime
import subprocess
import time
import functools
import random
import threading
import re
import traceback
from multiprocessing import Pool
from subprocess import check_output

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def watch(fn):
    fp = None

    while fp is None:
        if os.path.isfile(fn):
            fp = open(fn, "r")
        else:
            time.sleep(1)

    while True:
        new = fp.readline()
        if new:
            yield new
        else:
            time.sleep(1)


def tail_thread(fn):
    for line in watch(fn):
        print(line, end="")


def run(command, verbose=True, noop=False):
    """Print command then run command"""
    return_val = ""

    if verbose:
        print(command)
    if not noop:
        try:
            return_val = subprocess.check_output(
                command, shell=True, stderr=subprocess.PIPE
            ).decode()
        except subprocess.CalledProcessError as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}\n\n{e.returncode}\n\n{e.stdout.decode()}\n\n{e.stderr.decode()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise e
        except Exception as e:
            err_mesg = f"{os.getcwd()}: {e}\n\n{traceback.format_exc()}"
            print(err_mesg, file=sys.stderr)
            with open("err.txt", "w") as fd:
                fd.write(err_mesg)
            raise e
        if verbose and return_val:
            print(return_val)

    return return_val


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    import subprocess, os

    pipe = subprocess.Popen(
        "bash -c 'source %s > /dev/null; env'" % script,
        stdout=subprocess.PIPE,
        shell=True,
    )
    output = pipe.communicate()[0].decode()
    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)


def init_parser(parser):
    parser.add_argument("logs", metavar="Log Files", type=str, nargs="+")


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="""Print running slurm out until job completes."""
    )
    init_parser(parser)
    args = parser.parse_args()

    # Get slurm id.
    slurmid = None
    while slurmid is None:
        if os.path.isfile("slurm.job"):
            with open("slurm.job", "r") as fd:
                slurmid = fd.readline().strip()
        else:
            time.sleep(1)

    # Launch reading threads.
    threads = []
    for file in args.logs:
        tail = threading.Thread(target=tail_thread, args=(file,), daemon=True)
        tail.start()
        threads.append(threads)

    # Wait until slurm job is complete.
    while True:
        squeue = run("squeue", verbose=False)
        if slurmid in squeue:
            time.sleep(1)
        else:
            break

    # Wait for last output.
    time.sleep(1)


if __name__ == "__main__":
    main()
