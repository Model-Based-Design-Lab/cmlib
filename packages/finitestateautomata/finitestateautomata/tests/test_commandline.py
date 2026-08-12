"""
tests for the command line interface of the finitestateautomata package.
"""

import json
from pathlib import Path
from typing import Any, List
import pytest

from finitestateautomata.utils.commandline import main
from finitestateautomata.tests.utils import apply_regular_expressions, normalize

TEST_DIR = Path(__file__).parent
TESTCASE_FILE = TEST_DIR / "commandline" / "testcases.json"


def json_load(file_path):
    """Load JSON data from a file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


CASES = json_load(TESTCASE_FILE)


@pytest.mark.parametrize("case", CASES)
def test_commandline(case, monkeypatch, capsys):
    """ Test the command line interface of the finitestateautomata package using test cases
    defined in a JSON file."""

    args = ["finitestateautomata", *case["args"]]
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
    eliminate_whitespace = case.get("eliminate_whitespace", False)

    custom_regular_expressions: List[Any] = case.get("custom_regular_expressions", [])
    captured_output = apply_regular_expressions(captured.out, custom_regular_expressions)

    if "expected_output_file" in case:
        expected_output_file_path: Path = TEST_DIR / \
            "commandline" / case["expected_output_file"]
        with open(expected_output_file_path, encoding="utf-8") as f:
            expected_output = f.read()

        expected_output = apply_regular_expressions(
            expected_output, custom_regular_expressions)

        output_normalized = normalize(
            captured_output.strip(), eliminate_whitespace=eliminate_whitespace)
        expected_normalized = normalize(
            expected_output.strip(), eliminate_whitespace=eliminate_whitespace)
        result = output_normalized == expected_normalized
        assert result, (f"Expected output from file {case['expected_output_file']}, "
                        f"but got:\n {captured_output.strip()}")
    else:
        output_normalized = normalize(
            captured_output.strip(), eliminate_whitespace=eliminate_whitespace)
        expected_output = case["expected_output"]
        expected_output = apply_regular_expressions(
            expected_output, custom_regular_expressions)
        expected_normalized = normalize(
            expected_output.strip(), eliminate_whitespace=eliminate_whitespace)
        result = output_normalized == expected_normalized
        assert result, (f"Expected {case['expected_output']}, "
                        f"but got:\n {captured_output.strip()}")
