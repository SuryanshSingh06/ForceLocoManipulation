# Workflow, formats and history

## Environments observed

| Purpose | Local environment / repository |
| --- | --- |
| Initial Holosoma retargeting | `hsretargeting`, `~/holosoma` |
| HOI contact retargeting and resampling | `hoi-retarget`, `~/hoi-retarget` |
| Isaac geometry inspection | `hssim`, `~/holosoma` |

The supplied simulator inventory listed Isaac Sim 5.1.0.0, isaaclab package 0.47.2, Holosoma 0.0.1 and torch 2.7.0+cu128. These are reported versions from that inspection, not a portable lock file. The laptop uses Ubuntu and an RTX 5060 Laptop GPU with 8 GB VRAM. Do not assume all tools share one Python environment or that a later Newton installation changed these retargeting commands.

## T1 selection history relevant to lift

The T1 selector classified lift/carry/push/pull from source trajectories. This folder shares only lift results and the lift-only batch runner; the repository's pre-existing T1 file is not replaced.

| Run | Observed lift candidates | Issue / interpretation |
| --- | ---: | --- |
| run01 | 0 | 2,122 input errors, including conversion of tensors requiring gradients directly to NumPy |
| run02 | 3 | Input errors fixed; lift detection still overly restrictive |
| run03 | 0 | 2,122 input errors recurred |
| run04 | 14 | Version 1.1.0 lift segmentation improvements; 103 lift-invalid-skeleton diagnostic episodes |
| run05 | 117 | Overly restrictive skeleton validation removed; 0 input errors |

The run05 counts supplied were lift 117, carry 227, push 295, pull 206. Labels can overlap, so class sums need not equal distinct clip/recording counts. The skeleton issue was exposed by bent-over humans whose heads were temporarily below their pelvis, despite stable toe heights. That pose is not by itself an invalid skeleton.

Self-tests reported rotated-bottom handling, facing, lift/carry/push/pull, initially-held objects, raised chair, sideways interaction, ambiguous distance, reversal segmentation, lift-then-carry/lower and rejection of moving-rise/no-hold examples. Synthetic tests validate specific logic, not every dataset clip.

The later question about 103 candidates was not accompanied by a new authoritative manifest. Do not report 103 as the current total or infer its object-class breakdown from the 24-clip export inventory.

## Early 40-clip experiment

An earlier phase selected the 10 most recent manual acceptances in each of four classes. Eight previews were identified as bad: 1 lift, 0 carry, 3 push, 4 pull, leaving 32. The older NPZ export log contained 9 lift, 10 carry, 7 push and 6 pull. That mixed-class experiment is historical and is not this upload's selection authority.

The first lift-quality batch reported 5 completed, 9 quality-rejected and no failures. The user then requested screening all lift candidates rather than stopping at 10. The later selected-lift pipeline produced 24 exports. Those stages should not be conflated.

## Current selected lift inventory

The uploaded 2026-10-21 inventory contains 24 clips, all at 50 FPS, with zero export failures:

- smallbox: 6
- plasticbox: 13
- largebox: 5

The compact per-clip inventory is in `evidence/lift24_inventory_summary.json`. The 69 attempted output folders later found on the laptop are not 69 passed references. Use `passed_lift_manifest.json`, not a folder count.

## Cropping and generation

The batch runner expects the InterMimic tensor layout `(frames, 591)` and uses T1 `[start:end]` frame intervals with end exclusive. It detaches to CPU before NumPy conversion and stages a cropped `.pt` input for each selected lift. It records settings, source checksums and quality thresholds for resume checks. A reused output with incompatible settings should cause an explicit failure rather than silently overwriting reviews.

Lift source and retargeted quality checks operate on motion/object curves, with rotated object-bottom geometry and additional robot checks. `quality_rejected` is distinct from `failed`, `interrupted` and `completed`. Only completed records enter the passed manifest.

The G1 HOI path required SMPL-X model assets to reconstruct the human source skeleton. An initial download with keys such as `J_regressor`, `weights`, `v_template` lacked `hands_componentsl/r` and failed to load. The subsequently extracted full SMPL-X model files passed that key check. Keep these body models local and install them using their upstream distribution requirements; they are neither generated references nor included assets here.

## Data formats

| Format | Main fields | Use |
| --- | --- | --- |
| T1 JSON | IDs, task, source, start/end, object, review | Selection/provenance |
| Cropped source `.pt` | InterMimic tensor | HOI input, not a robot motion export |
| Early Holosoma `.npz` | `qpos`, `human_joints`, `fps`, `cost` | Initial robot/object retargeting result |
| HOI `.pkl` | root pose, 29 DOF positions, link poses, object pose, contact flags/targets, metadata | Contact optimization output |
| Holosoma training `.npz` | joint/body states and velocities, object states, names, FPS | Training-compatible reference data layout |
| MP4 | Rendered preview | Review only; no full motion state or generation proof |

The inspected HOI pickle has WXYZ root/object rotations, `dof_pos` width 29, and link poses for its own model. The export's `joint_pos` contains the 7-value free root plus 29 actuated joint positions; `joint_vel` contains 6 root velocities plus 29 joint velocities. Object position/quaternion/velocities are separate fields. Body counts depend on the exact model; preserve names and ordering rather than hard-coding the earlier 52-body example into every new asset.

NPZ means a ZIP-based container of named NumPy arrays. It is not a video. The earlier `viser_player.py` expected a `qpos` field; it cannot directly read the training export's `joint_pos` layout without an adapter.

## Resampling and scale

HOI's requested 50 FPS was capped at the 30 FPS source ceiling in the observed test. The batch therefore explicitly runs at 30 FPS, then resamples to 50 Hz. Positions are interpolated, orientations use quaternion interpolation, and discrete contacts are preserved with discrete sampling. Export recomputes model body poses and velocities.

The 41-frame HOI test spans `(41 - 1) / 30 = 1.333333 s`. A uniform 50 Hz grid that does not extrapolate produced 67 frames ending at 1.32 s. This slight endpoint truncation was intentional and disclosed. It is different from the earlier 42-frame Holosoma conversion to 69 frames.

The observed object geometry scale is approximately 0.83. The object model/trajectory must use matching conventions. Do not apply scale a second time in the simulator or replace an object model with an unscaled box just to hide a contact gap.

Only a largebox-specific combined Holosoma MJCF was initially present. The exporter was extended to produce per-clip scaled object models for smallbox/plasticbox as well, rather than require nonexistent `g1_29dof_w_smallbox.xml` etc.

## Watching and sharing

Render a contact pickle explicitly:

```bash
source ~/.holosoma_deps/miniconda3/etc/profile.d/conda.sh
conda activate hoi-retarget
cd ~/hoi-retarget
hoi-retarget-render \
  --input_pkl ~/holosoma/t2/hoi_test_lift/contact_window.pkl \
  --video_path ~/holosoma/t2/hoi_test_lift/contact_window.mp4 \
  --style studio --size 720 --headless
```

For batch rendering/sharing, use the IDs in the passed manifest and name videos by clip ID. There were initially zero `_review.mp4` videos at the checked location; another attempt counted 69 motion folders because it ignored the selection manifest. Neither observation means the 24 selected clips were all rendered and zipped successfully. No successful final video ZIP upload was supplied in this session.

VLC showed NVIDIA VDPAU/OpenGL surface-mapping errors and sometimes a Snap/GLIBC library conflict. A review launch used software decoding (`--avcodec-hw=none`) and X11 output (`--vout=xcb_x11`), with `LD_LIBRARY_PATH`/`LD_PRELOAD` removed where needed. Playback failure alone was not a retargeting failure.
