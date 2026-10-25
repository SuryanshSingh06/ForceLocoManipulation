# Carry retargeting: file structure and handoff

Verified against branch `carry-t1-inputs-20261025-202226` on October 25, 2026.

Repository: https://github.com/SuryanshSingh06/ForceLocoManipulation
Carry folder: https://github.com/SuryanshSingh06/ForceLocoManipulation/tree/carry-t1-inputs-20261025-202226/retargeting/carry

## 1. Start here

The source collection has **221 carry clips**: 88 largebox, 50 smallbox, and 83
plasticbox. Review state is **11 accepted and 210 pending**. These are human
reference segments, not 221 approved robot motions. Explicitly rejected T1
clips were excluded from this upload.

Use `retargeting/carry/t1/manifest.json` to locate recordings and crop frames.
Use `retargeting/carry/t1/object_assets/` for the original object geometry.
Install HOI-Retarget or Holosoma separately.

## 2. Actual files already on this branch

All paths below are relative to the repository root.

```text
retargeting/carry/
    README.md
    FILE_STRUCTURE.md                  This handoff guide
    .gitattributes                     Git LFS rules for large assets
    scripts/
        resolve_manifest.py            Converts portable paths to local absolute paths
        t1_select.py                   Snapshot of the T1 selection/review script
        upload_t1_carry.py              Uploader and provenance tooling
    t1/
        manifest.json                  All 221 carry IDs, sources, intervals, review states
        upload_inventory.json          Counts, file inventory and upload checksums
        source_recordings/
            <source-sha256-prefix>/
                <original-recording>.pt
        object_assets/
            largebox/
                largebox.obj
                largebox.urdf
            smallbox/
                smallbox.obj
                smallbox.urdf
            plasticbox/
                plasticbox.obj
                plasticbox.urdf

github_lift_carry/                      Older mixed lift/carry package, also on this branch
    README.md
    retained_t1_manifest.json           Source selections for the retained older package
    selected40_manifest.json           Earlier selection, before subsequent removals
    manifest.json                      Exported-package metadata
    hoi_selected40/
        carry/<clip-id>/               Older HOI results for 10 retained carry clips
        lift/<clip-id>/                Older lift results; not part of carry input collection
    motions/
        carry/                         Older carry training-format NPZ exports
        lift/
    previews/
        carry/                         Older carry videos
        lift/
    models/                            Older exported/scaled model assets
        assets/
    scripts/
        t2_batch_retarget.py
        t2_resample_hoi.py
        t2_export_holosoma.py
```

The older `github_lift_carry/` package originated on branch `t3-carry`.
Its 10 retained carry clips are 6 smallbox and 4 largebox. They are a separate
earlier selection, not retargeted outputs for all 221 source clips. Consult that
package's README and manifests before reusing its exports or scripts. Existing
lift-only runners under `retargeting/lift/`, if present on another branch,
must not be used unchanged for carry.

## 3. Download the actual assets, not Git LFS pointers

In a terminal with Git and Git LFS installed, clone to a new directory:

```bash
git clone --single-branch --branch carry-t1-inputs-20261025-202226 \
  https://github.com/SuryanshSingh06/ForceLocoManipulation.git \
  ForceLocoManipulation-carry
cd ForceLocoManipulation-carry
git lfs install
git lfs pull
git lfs fsck
python3 retargeting/carry/scripts/resolve_manifest.py \
  --out "$PWD/carry_manifest_local.json"
```

The resolver refuses to overwrite an existing output. Reuse that file if the
checkout has not moved, or choose a fresh output filename.

The small `.pt` and `.obj` blobs visible on GitHub may be LFS pointer text.
Downloading a repository ZIP alone does not establish that the full asset bytes
are available. `git lfs pull` must succeed; hashes should match the manifest.

## 4. Interpret the manifest correctly

Each `clips[]` row identifies one carry segment:

| Field | Meaning |
| --- | --- |
| `id` | Unique segment ID; use it as the output directory name |
| `source` | Full original recording, relative to the manifest's own directory |
| `source_id` | Original recording stem, useful for subject-dependent scaling |
| `start`, `end` | Source frame interval `[start:end]`, with end excluded |
| `object` | `largebox`, `smallbox`, or `plasticbox` |
| `review` | Original T1 review state, usually pending |
| `sha256` | Checksum of the full source recording, not the cropped segment |
| `source_original` | Historical uploader-machine path; do not use on a new machine |

Example real segment:

```text
id:       sub10_largebox_047_t1_carry_000049_000143
source:   source_recordings/0b592d3a4d283787/sub10_largebox_047.pt
start:    49
end:      143
frames:   94
object:   largebox
review:   pending
```

A runner must load the original source and crop `tensor[start:end]`. Passing
the full original recording directly to a retargeter will include actions
outside the chosen carry interval. Preserve the original source filename in a
per-clip input directory so tools that infer subject height from the name work.

The recording format used in this workflow is an InterMimic/OMOMO PyTorch tensor,
expected as `(T, 591)`; inspect and validate before batch processing. It contains
human/object trajectories, not a skinned-human triangle mesh. In T1, coordinates
are metres with Z up and source quaternions are XYZW. Robot-output quaternion
conventions must be checked separately. The source recordings used here are
30 FPS, but the portable manifest's `conventions.fps` is null: do not treat that
field as a verified per-file FPS value.

## 5. Prepare a local workspace

The following is a **recommended new layout**, not an existing uploaded dataset:

```text
carry_work/
    manifest_local.json
    selected_manifest.json
    assets/<object>/
    runs/<run-name>/
        settings.json
        summary.json
        carry/<clip-id>/
            selection.json
            input/<original-recording>.pt
            retarget.log
            status.json
            kinematic_window.pkl        HOI stage 1, when using HOI
            contact_window.pkl          HOI optimized result, when using HOI
            contact_window_fps50.pkl     Resampled HOI reference, if needed
            preview.mp4
            motion_original.npz         Alternative Holosoma native qpos output
            motion_fps50.npz             Alternative Holosoma qpos playback output
    exports/
        motions/carry/<clip-id>.npz      Validated Holosoma training schema
        models/
        manifest.json
```

Choose HOI or Holosoma explicitly and record the backend. These are alternative
retargeting paths; one run is not expected to produce both sets of native files.
Use one folder per clip ID, record errors, and resume completed clips by checking
their settings and output hashes. Do not overwrite original source assets.

## 6. Object assets and external dependencies

The raw meshes in `t1/object_assets/` are the source geometry. The supplied URDFs
use references such as `objects/smallbox/smallbox.obj` rather than simply
`smallbox.obj`. A standalone URDF loader resolving relative to the URDF directory
will not find that path in the uploaded layout. In a **staged copy**, resolve each
visual/collision mesh to the matching uploaded OBJ, or arrange the expected
directory layout. Keep mesh scale and coordinate origins unchanged. Do not
blindly substitute the older scaled meshes from `github_lift_carry/models/`.

A new machine still needs:

- The chosen retargeting framework and its Python environment.
- The standard G1 robot assets and joint mapping expected by that framework.
- Licensed SMPL-family body-model files if the chosen pipeline requires them;
  these weights are not included in this repository.
- A carry-capable crop/batch runner. The upload and manifest resolver do not
  perform retargeting.
- For Holosoma, its `demo_data/height_dict.pkl` available relative to the runtime
  working directory, plus a valid G1/object combined MJCF for each object.
  The original installation used here only had a largebox combined MJCF, so
  smallbox/plasticbox models required staging and validation.

## 7. Retargeting settings

For **HOI-Retarget**, stage one cropped segment first. Example with replaceable
paths, not a command to run against an uncropped full recording:

```bash
hoi-retarget \
  --input_file /absolute/path/to/clip/input/sub10_largebox_047.pt \
  --out_dir /absolute/path/to/run/carry/sub10_largebox_047_t1_carry_000049_000143 \
  --robot unitree_g1 \
  --mode contact \
  --object_model_path /absolute/path/to/staged/largebox.urdf \
  --tgt_fps 30 \
  --no-record_video
```

`--mode contact` runs contact optimization after kinematic retargeting. Check the
saved metadata and logs and review `contact_window.pkl`, not just the kinematic
stage. This workflow's HOI tool limits retargeting to the source rate; requesting
50 FPS does not necessarily produce 50 FPS output. If required, resample the
result to 50 Hz afterward with quaternion-aware interpolation.

For **Holosoma**, the corresponding CLI choices are
`--task-type object_interaction --robot g1 --data-format smplh`, with the correct
object configured. `robot_only` is not the intended task for these interactions.
Holosoma's task-type flags and HOI's mode flags are different APIs.

## 8. Review and export before training

Keep T1 selection review separate from robot-motion review. A pending T1 clip
can be retargeted experimentally, but it is not automatically approved.

Check the robot carries the object with consistent hand contact, sensible object
scale, no obvious penetration, plausible joints, smooth motion, and reasonable
foot support. Optimization completion alone does not prove the grasp supports a
physical object. Log rejections with clip IDs and reasons.

A `qpos` playback NPZ is not the same as a Holosoma training NPZ. For training,
use a validated exporter that supplies the required joint/body/object positions,
orientations, velocities, FPS, and names in the simulator's expected ordering.
Preserve provenance, backend settings, object scale, and review decisions in an
export manifest.

Share the source manifest, scripts/configuration, logs, approved output motions,
matching robot/object assets, previews, and environment/version information.
Policy checkpoints and training results belong to a later training handoff.

