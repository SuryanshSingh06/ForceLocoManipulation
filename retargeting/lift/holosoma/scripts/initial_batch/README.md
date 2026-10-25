# Holosoma lift retargeting: all candidates and matching selected 24

This runner invokes the installed **Holosoma** `examples/robot_retarget.py` using:

```text
--task-type object_interaction
--robot g1
--data-format smplh
--task-config.object-name <each clip's object>
--retargeter.activate-obj-non-penetration
--retargeter.activate-joint-limits
--retargeter.activate-foot-sticking
```

It does not invoke HOI-Retarget and does not reuse HOI robot trajectories. The HOI passed manifest is read only to recover the exact same selected 24 clip IDs/frame ranges.

## Run on the laptop

Extract this folder into `~/holosoma/t2`, then:

```bash
bash ~/holosoma/t2/holosoma_lift_batch/run_holosoma_lifts.sh
```

The launcher activates `hsretargeting`, the environment used for the earlier successful Holosoma test. It does not change the Isaac/Newton simulation environment or robot setup.

Defaults:

| Input / output | Path under `~/holosoma` |
| --- | --- |
| T1 selections | `t1/run05/manifest.json` |
| Exact selected 24 | `t2/hoi_all_lift_candidates01/passed_lift_manifest.json` |
| All new Holosoma lifts | `t2/holosoma_lift_all01` |
| Independent copies of selected 24 | `t2/holosoma_lift_selected24_01` |

By default, **all pending and accepted T1 lift clips** are attempted; explicitly T1-rejected clips are excluded. There is no 10-clip limit and no HOI quality-pass filter on the all-clips run. If you literally want rejected T1 lifts as well, add `--include-rejected` to the launcher. No other action class is included.

Each selected-24 ID must also be in the chosen all-lift selection. Missing/rejected IDs or different source frame ranges fail before retargeting rather than silently changing the comparison set.

For a no-compute selection check:

```bash
bash ~/holosoma/t2/holosoma_lift_batch/run_holosoma_lifts.sh --dry-run
```

The script prints exact input counts and object breakdown, so the run does not assume that all 103 or all 117 historical candidates are still eligible.

## Outputs

Each output collection has a `manifest.json`. Successful per-clip folders are `lift/<clip_id>/` and contain:

- `motion_original.npz`: native Holosoma qpos result with saved FPS, 43 coordinates per frame.
- `motion_fps50.npz`: uniform 50 Hz **playback** version, with quaternion interpolation for both root and object.
- `status.json`, `selection.json`, `retarget.log`: generation command, fingerprints, source interval and result status.

The all-lifts directory additionally stores the staged source crops and the retargeter's original-named results. The selected24 directory contains actual independent copies of successful matching results, not links to the all-lifts files and not a second optimization run. At the end, private model snapshots are available under `runtime/models` in both collections. Final manifests explicitly count failed/missing subset results; a failed clip is not presented as one of 24 successful clips.

Run the same command to resume. A completed result is reused only when source, crop, run settings and saved NPZ hashes match. Interrupted/failed jobs are retried. Changed script/settings or conflicting output contents require fresh sibling output paths, for example:

```bash
bash ~/holosoma/t2/holosoma_lift_batch/run_holosoma_lifts.sh \
  --all-out ~/holosoma/t2/holosoma_lift_all02 \
  --selected-out ~/holosoma/t2/holosoma_lift_selected24_02
```

Keep the terminal running until completion. Ctrl+C stops the current subprocess and records interruption for resumption. The run returns a nonzero exit code if some clips failed; inspect their logs and rerun after fixing the cause.

## Standard robot and object assets

The installed `models/` folder is copied into a private runtime. Standard `models/g1/g1_29dof.urdf` and the standard combined G1/object XMLs are used by the G1 configuration; no Isaac half-sphere preset is selected or installed by this runner.

The earlier local installation had only `g1_29dof_w_largebox.xml`. Where missing, the runner generates smallbox/plasticbox combined XMLs in the private copy by retaining the template's robot and replacing the free object body/mesh. It preserves standard 29-joint kinematics. The raw object URDF and its referenced mesh must already be present in Holosoma or in `~/hoi-retarget/assets/objects`. The resolver searches nested folders and follows the mesh filename in the URDF, rather than assuming a sibling smallbox.obj. A non-OBJ visual is converted to a canonical OBJ with no scale change. These are unscaled source geometry assets, not the 0.83 HOI-export geometry. Holosoma retains its own native scaling/preprocessing conventions.

If a raw object URDF has nonzero visual transforms or a nonunit mesh scale, adaptation stops for explicit frame verification. Generated object inertial parameters remain the template values because this is kinematic retargeting, not a validated dynamics model. Do not use those placeholder object dynamics to claim force-based grasp validation.

The runner checks the installed CLI's help before using the explicit constraint flags, then loads each combined model with MuJoCo and verifies `nq=43` before the long batch. It does not patch installed Holosoma source/models. Local upstream code can differ from the source reviewed while building this helper; unrecognized flags/layouts cause a preflight error.

## Playback versus training export

These are Holosoma retargeter `qpos` NPZ outputs. The optional 50 Hz version is compatible with qpos playback layouts and does **not** contain the final training-schema `joint_pos/body_pos_w/...` arrays and velocities. That conversion is a separate step after comparing/approving these new results. A 50 Hz uniform grid can end less than one 20 ms step before the final source time to avoid extrapolation.

Object-interaction retargeting and nonpenetration constraints do not prove a physically supportable grasp or enforce exact surface contact in every frame. Review the new results before using them as training references.

## Local verification of this helper

Seven tests passed for lift selection, selected-ID/range consistency, quaternion-aware resampling, private object-model adaptation and checksum-checked independent subset copies. Python syntax and Bash syntax were checked. Full optimization was not run in the coding environment, which does not have the laptop's Holosoma solver, torch environment or MuJoCo installation. The first real execution is on the laptop.
