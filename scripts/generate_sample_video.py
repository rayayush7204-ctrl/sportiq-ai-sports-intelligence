"""Utility script to generate a synthetic football match clip in data/raw/sample.mp4."""

import sys
from pathlib import Path

import cv2
import numpy as np


def generate_sample_match_clip(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration_seconds: float = 3.0,
) -> Path:
    """Generate a realistic synthetic football clip with pitch markings and players."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(fps * duration_seconds)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Simulated player positions (x, y, team_color)
    players = [
        {"x": 300, "y": 360, "vx": 2.0, "vy": 0.5, "color": (255, 0, 0)},     # Team A (Blue)
        {"x": 450, "y": 250, "vx": 1.5, "vy": -0.8, "color": (255, 0, 0)},
        {"x": 600, "y": 420, "vx": 1.0, "vy": 1.2, "color": (255, 0, 0)},
        {"x": 750, "y": 300, "vx": -1.8, "vy": 0.4, "color": (0, 0, 255)},   # Team B (Red)
        {"x": 900, "y": 450, "vx": -1.2, "vy": -1.0, "color": (0, 0, 255)},
        {"x": 1050, "y": 350, "vx": -2.0, "vy": 0.2, "color": (0, 0, 255)},
    ]

    ball = {"x": 550.0, "y": 360.0, "vx": 3.5, "vy": 0.3}

    for frame_idx in range(total_frames):
        # 1. Pitch background (grass green)
        frame = np.full((height, width, 3), (45, 120, 35), dtype=np.uint8)

        # 2. Pitch lines (white markings)
        # Touchlines & goal lines
        cv2.rectangle(frame, (80, 50), (width - 80, height - 50), (255, 255, 255), 3)
        # Halfway line
        cv2.line(frame, (width // 2, 50), (width // 2, height - 50), (255, 255, 255), 3)
        # Center circle
        cv2.circle(frame, (width // 2, height // 2), 90, (255, 255, 255), 3)
        cv2.circle(frame, (width // 2, height // 2), 5, (255, 255, 255), -1)

        # 3. Draw and update players
        for p in players:
            px = int(p["x"] + p["vx"] * frame_idx)
            py = int(p["y"] + p["vy"] * frame_idx)
            # Torso
            cv2.rectangle(frame, (px - 14, py - 30), (px + 14, py + 15), p["color"], -1)
            # Head
            cv2.circle(frame, (px, py - 38), 10, (220, 200, 180), -1)
            # Legs
            cv2.line(frame, (px - 7, py + 15), (px - 10, py + 40), (40, 40, 40), 4)
            cv2.line(frame, (px + 7, py + 15), (px + 10, py + 40), (40, 40, 40), 4)

        # 4. Draw and update ball
        bx = int(ball["x"] + ball["vx"] * frame_idx)
        by = int(ball["y"] + ball["vy"] * frame_idx + np.sin(frame_idx / 4.0) * 8)
        cv2.circle(frame, (bx, by), 9, (255, 255, 255), -1)
        cv2.circle(frame, (bx, by), 9, (0, 0, 0), 1)

        # 5. Timestamp and frame overlay
        timestamp_sec = frame_idx / fps
        cv2.putText(
            frame,
            f"SportIQ Synthetic Feed | Frame: {frame_idx:04d} | Time: {timestamp_sec:.2f}s",
            (90, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    print(f"Generated synthetic football video: {output_path} ({total_frames} frames, {width}x{height}, {fps} FPS)")
    return output_path


if __name__ == "__main__":
    target = Path("data/raw/sample.mp4")
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    generate_sample_match_clip(target)
