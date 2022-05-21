import argparse

def _str_to_bool(s: str):
    """
    Convert string to bool (in argparse context).

    """
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

# https://stackoverflow.com/a/36194213/11764120
def add_boolean_argument(parser: argparse, name: str, default: bool = False, messages: str = ""):
    """
    Add a boolean argument to an ArgumentParser instance.

    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool, help=messages)
    group.add_argument('--no' + name, dest=name, action='store_false', help=messages)