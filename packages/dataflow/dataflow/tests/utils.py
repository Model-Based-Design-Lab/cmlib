""" Utilities for testing. """

import re


def normalize(s, eliminate_whitespace=False):
    """Normalize whitespace in a string by replacing sequences of whitespace with
    a single space and stripping leading/trailing whitespace."""

    if eliminate_whitespace:
        return re.sub(r'\s+', '', s.strip())
    return re.sub(r'\s+', ' ', s.strip())


def apply_regular_expressions(s, regex_list):
    """Apply a list of regular expressions to a string."""
    for regex in regex_list:
        s = re.sub(regex["pattern"], regex["replacement"], s)
    return s