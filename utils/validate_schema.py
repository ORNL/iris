#!/usr/bin/env python3

import argparse
from jsoncomment import JsonComment
import jsonschema
from jsonschema import validate, Draft7Validator
import subprocess
import sys

json = JsonComment()

def validate_json_data(json_data, schema):
    validate(instance=json_data, schema=schema)

def validate_schema(schema):
    Draft7Validator.check_schema(schema)

def InitParser(parser):
    # Todo: Change the parser to expose the single task variables.
    parser.add_argument('-i', '--input', help='Input JSON file', type=str, required=True)
    parser.add_argument('-s', '--schema', help='Input JSON Schema file', type=str, required=True)

def run_task(argv):
    """Parse the arguments and call execute."""
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Description""")
    InitParser(parser)
    args = parser.parse_args(argv[1:])

    with open(args.input, 'r') as fd:
        json_data = json.load(fd)

    with open(args.schema, 'r') as fd:
        json_schema = json.load(fd)

    validate_schema(json_schema)

    validate_json_data(json_data, json_schema)

if __name__ == '__main__':
    run_task(sys.argv)
