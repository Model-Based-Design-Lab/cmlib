""" Utilities for testing. """

import re


def normalize(s):
    """Normalize whitespace in a string by replacing sequences of whitespace with
    a single space and stripping leading/trailing whitespace."""
    return re.sub(r'\s+', ' ', s.strip())
