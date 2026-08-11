# test_dataflow_commandline.py
import json
from unittest import case
import pytest
from pathlib import Path
from dataflow.utils.commandline import main

TEST_DIR = Path(__file__).parent
TESTCASE_FILE = TEST_DIR / "commandline" / "testcases.json"

with open(TESTCASE_FILE, encoding="utf-8") as f:
    CASES = json.load(f)

@pytest.mark.parametrize("case", CASES)
def test_commandline(case, monkeypatch, capsys):
    model_file_path: Path = TEST_DIR / "models" / case["model"]
    args = ["commandline.py", *case["args"]]
    args.append(str(model_file_path))
    monkeypatch.setattr("sys.argv", args)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == case["expected_exit_code"], f"Expected exit code {case['expected_exit_code']}, but got {excinfo.value.code}"

    captured = capsys.readouterr()

    if "expected_output_file" in case:
        expected_output_file_path: Path = TEST_DIR / "commandline" / case["expected_output_file"]
        with open(expected_output_file_path, encoding="utf-8") as f:
            expected_output = f.read()
        result = captured.out.strip().splitlines() == expected_output.strip().splitlines()
        assert result, f"Expected output from file {case['expected_output_file']}, but got {captured.out.strip()}"
    else:
        result = captured.out.strip().splitlines() == case["expected_output"].splitlines()
        assert result, f"Expected {case['expected_output']}, but got {captured.out.strip()}"
