import sys
from unittest.mock import patch

import pytest

from .. import data
from ..cli import run


def test_cli_integration(tmp_path):
    bids_dir = data.load("tests/ds000005")
    out_dir = tmp_path / "out"
    fs_license = tmp_path / "license.txt"
    fs_license.write_text("dummy")

    argv = [
        "petprep",
        str(bids_dir),
        str(out_dir),
        "participant",
        "--fs-license-file",
        str(fs_license),
        "--skip-bids-validation",
        "--nthreads",
        "1",
        "--omp-nthreads",
        "1",
    ]

    with patch.object(sys, "argv", argv), patch(
        "nipype.pipeline.engine.Workflow.run", return_value=None
    ) as run_patch:
        with pytest.raises(SystemExit) as excinfo:
            run.main()

    assert excinfo.value.code == 0
    run_patch.assert_called_once()

    petprep_dir = out_dir / "petprep"
    assert petprep_dir.exists()
    log_root = petprep_dir / "sub-01" / "log"
    assert log_root.exists() and any(log_root.iterdir())