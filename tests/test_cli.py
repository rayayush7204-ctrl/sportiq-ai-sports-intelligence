"""Integration tests for sportiq CLI."""

import json
from pathlib import Path

from sportiq.cli import main


def test_cli_check_env(capsys):
    """Verify 'sportiq check-env' executes and reports runtime environment."""
    exit_code = main(["check-env", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert "sportiq_version" in data
    assert "python_version" in data
    assert "opencv_version" in data
    assert "torch_version" in data
    assert data["cuda_available"] is True


def test_cli_inspect_video(synthetic_video_path: Path, capsys):
    """Verify 'sportiq inspect-video' reports correct video properties."""
    exit_code = main(["inspect-video", str(synthetic_video_path), "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    meta = json.loads(captured.out)

    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 30.0
    assert meta["total_frames"] == 60


def test_cli_inspect_missing_video(tmp_path: Path, capsys):
    """Verify 'sportiq inspect-video' handles missing files gracefully."""
    missing = tmp_path / "missing.mp4"
    exit_code = main(["inspect-video", str(missing)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err or "not found" in captured.err


def test_cli_extract_frames(synthetic_video_path: Path, tmp_path: Path, capsys):
    """Verify 'sportiq extract-frames' runs via CLI with flags."""
    out_dir = tmp_path / "cli_extract"
    exit_code = main([
        "extract-frames",
        str(synthetic_video_path),
        "--output-dir",
        str(out_dir),
        "--every-seconds",
        "1.0",
        "--json",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert summary["extracted_frames_count"] == 2
    assert (out_dir / "frames.jsonl").exists()
    assert (out_dir / "video_metadata.json").exists()
