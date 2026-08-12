"""
tests for the command line interface of the markovchains package.
"""

import argparse
import json
from pathlib import Path
import pytest

from markovchains.utils.commandline import main
from markovchains.tests.utils import normalize

TEST_DIR = Path(__file__).parent
TESTCASE_FILE = TEST_DIR / "commandline" / "testcases.json"


def json_load(file_path):
    """Load JSON data from a file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


CASES = json_load(TESTCASE_FILE)


@pytest.mark.parametrize("case", CASES)
def test_commandline(case, monkeypatch, capsys):
    """ Test the command line interface of the markovchains package using test cases
    defined in a JSON file."""

    args = ["markovchains", *case["args"]]
    if "model" in case:
        model_file_path: Path = TEST_DIR / "models" / case["model"]
        args.append(str(model_file_path))
    monkeypatch.setattr("sys.argv", args)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == case[
        "expected_exit_code"], (f"Expected exit code {case['expected_exit_code']}, "
                                f"but got:\n {exc_info.value.code}")

    captured = capsys.readouterr()
    eliminate_whitespace=case.get("eliminate_whitespace", False)

    if "expected_output_file" in case:
        expected_output_file_path: Path = TEST_DIR / \
            "commandline" / case["expected_output_file"]
        with open(expected_output_file_path, encoding="utf-8") as f:
            expected_output = f.read()
        output_normalized = normalize(captured.out.strip(), eliminate_whitespace=eliminate_whitespace)
        expected_normalized = normalize(expected_output.strip(), eliminate_whitespace=eliminate_whitespace)
        result = output_normalized == expected_normalized
        import argparse
        print("argparse version:", argparse.__version__)
        assert result, (f"Expected output from file {case['expected_output_file']}, "
                         f"but got:\n {captured.out.strip()}")
    else:
        output_normalized = normalize(captured.out.strip(), eliminate_whitespace=eliminate_whitespace)
        expected_normalized = normalize(case["expected_output"].strip(), eliminate_whitespace=eliminate_whitespace)
        result = output_normalized == expected_normalized
        assert result, (f"Expected {case['expected_output']}, "
                         f"but got:\n {captured.out.strip()}")
