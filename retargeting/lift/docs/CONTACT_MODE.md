# Contact mode and the complete generation commands

## What was actually used

Two different retargeting tools were used at different stages. Their flags are not interchangeable.

### 1. Initial Holosoma object-interaction test

This command generated an earlier G1 `qpos` NPZ for the cropped test lift:

```bash
source ~/.holosoma_deps/miniconda3/etc/profile.d/conda.sh
conda activate hsretargeting
cd ~/holosoma/src/holosoma_retargeting/holosoma_retargeting

python examples/robot_retarget.py \
  --task-type object_interaction \
  --robot g1 \
  --data-format smplh \
  --task-name sub10_largebox_032 \
  --data-path ~/holosoma/t2/test_lift/input \
  --task-config.object-name largebox \
  --save-dir ~/holosoma/t2/test_lift/retargeted
```

The reported result was 42 frames, `qpos` shape `(42, 43)`, at 30 FPS. `43 = 7 robot root + 29 robot joints + 7 object pose`. The initial conversion produced 69 frames at 50 FPS. This is the early Holosoma route, not the later 41-frame HOI test output.

Holosoma's relevant task type was `object_interaction`. The `robot_only` setting was not the setting shown in this test command. Its CLI defines `robot_only`, `object_interaction` and `climbing`.

### 2. HOI-Retarget single-clip test

This is the HOI test command shown in the session, including the requested 50 FPS:

```bash
source ~/.holosoma_deps/miniconda3/etc/profile.d/conda.sh
conda activate hoi-retarget
cd ~/hoi-retarget

hoi-retarget \
  --input_file ~/holosoma/t2/test_lift/input/sub10_largebox_032.pt \
  --out_dir ~/holosoma/t2/hoi_test_lift \
  --robot unitree_g1 \
  --mode contact \
  --tgt_fps 50 \
  --no-record_video
```

The object model was inferred as `~/hoi-retarget/assets/objects/largebox.urdf`; auto gender resolved to neutral. The help supplied in-session showed headless rendering enabled by default. Later suggested commands made those defaults explicit using `--object_model_path`, `--gender neutral`, and `--headless`; those were equivalent suggestions, not evidence of the literal original invocation.

The log showed:

```text
Window projection (mode=contact): q-only contact NLP.
--tgt_fps 50 requested; the clip is 30 fps (the source rate is the ceiling).
Saved kinematic_window: .../kinematic_window.pkl
Unified-window pass complete: 1 windows, 0 failed
Saved contact_window: .../contact_window.pkl
```

The inspected HOI pickle had 41 frames at 30 FPS. Why that HOI path yielded 41 rather than the 42 input frames was not investigated to completion. Do not treat 41 and 42 as identical or overwrite one result with the other.

### 3. Lift quality-screening batch

The recovered batch script explicitly constructs this argument list for each cropped source recording:

```text
hoi-retarget --input_file <clip_folder>/input/<source_name>.pt
            --out_dir <clip_folder>
            --robot unitree_g1
            --mode contact
            --tgt_fps 30
            --no-record_video
```

Use the complete Bash wrapper at `scripts/run_lift_batch.sh` to reproduce the batch selection. The runner's defaults attempt only 10 accepted clips; the wrapper uses `--include-pending --limit 0` for the all-lift-candidate scan.

The batch checks the subprocess exit status, a logged unified-window completion with zero failed windows, both output pickle files, then automated retargeted curve quality. Its `completed` state means those checks passed, not that a physical grasp passed. It writes the exact per-clip command into `status.json`.

## How to verify the flag

For the selected IDs, inspect:

- `status.json`: the recorded command contains `--mode contact` and `--robot unitree_g1`.
- `retarget.log`: contact-mode setup and successful window passes.
- `contact_window.pkl`: `meta.projection_mode == 'contact'`, `meta.stage == 'contact_window'`.
- Preview source: render the contact result rather than `kinematic_window.pkl`.

The included `audit_lift_contact.py` automates those checks for trusted local pickles. File presence alone is weaker evidence than checking metadata plus commands/logs. A copied, stale or failed-run pickle can exist in a folder that is not part of the passed selection.

## What contact mode does and does not establish

The inspected HOI code used the cost

```python
contact_weight * sumsqr(p_object + R_object @ p_contact_object - p_robot_contact_link)
```

for each active contact link. The reported default `contact_weight` was 1000, with separate orientation pins and IK tracking weight. This is a weighted objective, so calling it contact optimization does not mean that visible mesh surfaces are subject to an exact zero-gap constraint.

Contact gating determines which frames/links receive this cost. A link can be inactive even while the object is airborne. Small marker-to-target error also does not guarantee that a different hand collider in Isaac touches the object.

In this q-only mode the object reference is scaled/smoothed preprocessing data and its pose is not optimized jointly with the robot. No physical load-bearing guarantee follows from the object rising in a replay.

## Upstream references

- [HOI-Retarget](https://github.com/leggedrobotics/hoi-retarget)
- [Holosoma retargeting README](https://github.com/amazon-far/holosoma/tree/main/src/holosoma_retargeting)
- [Holosoma robot_retarget.py](https://github.com/amazon-far/holosoma/blob/main/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py)

The linked repositories may evolve. These notes describe the help, source snippets, logs and recovered helper versions observed in this project; exact upstream commit hashes from the laptop were not supplied.
