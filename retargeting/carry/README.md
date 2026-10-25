# Carry inputs at the end of T1, before retargeting

This collection contains original human-motion `.pt` recordings, carry-only clip labels and frame intervals, raw object meshes, and available local T1 selection code. These are the inputs used to view the human skeleton and to retarget again, not HOI/robot motion outputs or policy checkpoints.

The scope and accepted/pending counts are recorded in `t1/upload_inventory.json`. Pending clips have not been manually approved. Full source recordings are copied once per unique recording/name; `[start:end]` selects each carry segment, with end exclusive. A full recording can contain actions outside the selected interval. Source filenames and bytes are preserved for subject-dependent retargeting and checksum verification.

Motion `.pt` tensors contain skeleton/object trajectories; they are not triangle meshes of a skinned human. Object meshes are supplied separately in `t1/object_assets`. SMPL-family body-model weights and complete third-party tool installations are not included.

## Download and use

Clone this branch with Git LFS and run `git lfs pull`. The committed manifest uses paths relative to its own directory. Convert it to local absolute paths before using older scripts that resolve paths against the working directory:

```bash
python3 retargeting/carry/scripts/resolve_manifest.py
```

That writes `~/Downloads/t1_carry_manifest_local.json`. Use a fresh `--out` if it already exists. Retarget with the separately installed HOI/Holosoma tool and the appropriate G1/object assets. The newer lift-only batch script intentionally rejects carry inputs, so use a carry-capable runner.

The earlier already-retargeted 10 carry clips remain under `github_lift_carry/` on the `t3-carry` branch. This folder supplies the pre-retargeting inputs and preserves review state. Checksums verify copied files; they do not certify motion or physical grasp quality.
