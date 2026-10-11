#!/usr/bin/env python3
"""Validate the T3 carry motion set without importing Isaac Sim or NumPy."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile


REQUIRED_ARRAYS = {
    "fps.npy",
    "joint_pos.npy",
    "joint_vel.npy",
    "body_pos_w.npy",
    "body_quat_w.npy",
    "body_lin_vel_w.npy",
    "body_ang_vel_w.npy",
    "object_pos_w.npy",
    "object_quat_w.npy",
    "object_lin_vel_w.npy",
    "object_ang_vel_w.npy",
    "joint_names.npy",
    "body_names.npy",
}


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    motion_dir = project_root / "t3/carry/motions/smallbox"
    manifest_path = project_root / "github_lift_carry/manifest.json"

    manifest = json.loads(manifest_path.read_text())
    expected = {
        Path(clip["npz"]).name
        for clip in manifest["clips"]
        if clip["task"] == "carry" and clip["object"] == "smallbox"
    }
    motions = sorted(motion_dir.glob("*.npz"))
    actual = {path.name for path in motions}

    errors: list[str] = []
    if actual != expected:
        errors.append(f"motion set mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if len(motions) != 6:
        errors.append(f"expected 6 small-box carry clips, found {len(motions)}")

    for path in motions:
        if not path.exists():
            errors.append(f"broken motion link: {path}")
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
        except zipfile.BadZipFile:
            errors.append(f"not a readable NPZ archive: {path}")
            continue
        missing = REQUIRED_ARRAYS - members
        if missing:
            errors.append(f"{path.name}: missing arrays {sorted(missing)}")

    if errors:
        print("Carry data validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Carry data validation passed: {len(motions)} small-box clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

