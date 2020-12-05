"""
general parser function
"""
import sys
import argparse

def parser(*args):
    """parser for command line arguments
       based on the first letter of the
       input names the input structure is
       defined
    Returns:
        [type]: parser with arguments
    """
    ARGUMENTS = sys.argv[1:]
    assert len(ARGUMENTS) != 0, "No command line input!"

    PARSER = argparse.ArgumentParser()
    first_chars = [a[0] for a in args]
    assert len(set(first_chars)) == len(first_chars), \
        "Params are not allowed to have the first starting letter!"
    for arg in args:
        short = "-" + arg[0]
        long = "--" + arg
        PARSER.add_argument(short, long)

    ARGS = PARSER.parse_args()

    return ARGS