from configargparse import ArgumentParser
import argparse

def add_default_parser_args(parser, query=False):
    parser.add("--cuda", type=bool, default=True, required=False, help="use cuda if available")
    parser.add("--precision", type=str, default='float32', required=False, help="the float-point precision used")
    return parser