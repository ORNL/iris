#!/usr/bin/env python3

"""
Adapted from dagger_runner.cpp
"""
__author__ = "Aaron Young, Beau Johnston"
__copyright__ = "Copyright (c) 2020-2023, Oak Ridge National Laboratory (ORNL) Programming Systems Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "1.0"

import iris
import argparse
import re
import sys
import dagger_generator as dg

def init_parser(parser):
    pass

def main(argv):
    # Parse the arguments
    args = dg.parse_args(additional_arguments=[init_parser])

    print(args)
    print (dg._graph,dg._depth,dg._num_tasks,dg._min_width,dg._max_width)

    exit(0)
    graph = iris.graph()
    graph.load('graph.json', params)


if __name__ == '__main__':
    main(sys.argv)
