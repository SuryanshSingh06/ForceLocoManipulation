# Lift retargeting: OMOMO to Unitree G1

This folder documents and preserves the **lift-only** reference-generation work. The main pipeline is **T1 selection → cropped OMOMO input → HOI contact optimization → 50 Hz resampling → Holosoma NPZ export → review**.

**Current status:** 24 lift exports are recorded in the uploaded inventory: **6 smallbox, 13 plasticbox, 5 largebox**. Export succeeded with no reported failures. These are candidate references, not 24 physically validated grasps. Hand contact remains unresolved. The repository includes code and diagnostic evidence; the large motion files and videos still live on Rudra's laptop until the uploader below is run.

## Start here for a review

1. Read [contact mode and commands](docs/CONTACT_MODE.md). This resolves the `--task-type robot_only` question.
2. Inspect [t2_batch_retarget.py](scripts/t2_batch_retarget.py). Its subprocess command explicitly supplies `--robot unitree_g1 --mode contact --tgt_fps 30 --no-record_video`.
3. Read [contact diagnostics](docs/CONTACT_DIAGNOSTICS.md). Small optimizer marker errors did not imply that the simulation hand surfaces touched the object.
4. Use [upload_lift_assets.py](scripts/upload_lift_assets.py) on the laptop to publish the actual selected outputs, generation commands/logs, plots and assets.

## Tools used, without conflating their flags

| Route | Robot | Interaction/contact selection | Output |
| --- | --- | --- | --- |
| Earlier Holosoma retargeting test | `--robot g1` | `--task-type object_interaction` | `qpos` NPZ |
| Later HOI-Retarget pipeline | `--robot unitree_g1` | `--mode contact` | `kinematic_window.pkl`, `contact_window.pkl` |

`--task-type robot_only` belongs to Holosoma's retargeting CLI. It is not an option in the installed HOI CLI help supplied during this work. Specifying `unitree_g1` chooses an embodiment; it does not by itself confirm that HOI, exported MuJoCo and training URDF hand geometry agree.

## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/t2_batch_retarget.py` | Lift-only cropping, curve quality screening, explicit HOI contact runs and resumable statuses |
| `scripts/run_lift_batch.sh` | Complete Bash wrapper for screening all nonrejected lift candidates |
| `scripts/t2_resample_hoi.py` | Resample saved HOI output to a uniform 50 Hz grid |
| `scripts/t2_palm_geometry_qa.py` | Lift NPZ preparation, inventory and calibrated geometry QA |
| `scripts/t2_export_holosoma.py` | Model/export helper used by preparation; its standalone CLI retains earlier 32-clip defaults, so use `prepare` below for lift-only exports |
| `scripts/audit_lift_contact.py` | Audit selected outputs' saved mode, command and optimization log |
| `scripts/upload_lift_assets.py` | Collect selected lift artifacts and optionally push an isolated upload branch |
| `scripts/t2_isaac_contact_view.py` | Paused Isaac geometry inspection, not dynamic grasp validation |
| `scripts/isaac_*`, `scripts/t2_run_dynamic_with_contacts.py` | Diagnostic helpers; dynamic wrapper needs an actual evaluation entry point |
| `evidence/` | Uploaded pose/collider snapshots, export inventory summary and unfilled calibration template |
| `docs/` | Workflow, limitations, provenance and review guidance |
| `artifacts/<dataset>/` | Laptop-generated upload payload, ignored locally and explicitly added by the uploader |

## Run the lift batch

Existing setup on the laptop: `~/holosoma`, `~/hoi-retarget`, Conda environment `hoi-retarget`. Source inputs are the OMOMO/InterMimic `.pt` recordings already installed there. Body-model assets must be obtained and installed separately; they are not included here.

From this repository:

```bash
bash retargeting/lift/scripts/run_lift_batch.sh
```

The wrapper selects **lift only**, includes pending T1 candidates, excludes rejections and uses `--limit 0` to remove the runner's default limit of 10. The existing run directory supports guarded resumption; a changed script/config may require a fresh `--out` rather than overwriting reviews. Do not count every output folder as accepted: an attempted/rejected clip can retain generated files.

The runner writes `passed_lift_manifest.json` containing only completed, quality-passing clips. Preserve it as the downstream selection authority. Curve screening is an automated heuristic, not a contact or physics certificate.

## Convert the selected lifts to 50 Hz Holosoma references

Run in `hoi-retarget` from this repository:

```bash
source ~/.holosoma_deps/miniconda3/etc/profile.d/conda.sh
conda activate hoi-retarget
python retargeting/lift/scripts/t2_palm_geometry_qa.py prepare \
  --batch ~/holosoma/t2/hoi_all_lift_candidates01 \
  --out ~/holosoma/t2/lift24_npz_contactqa
```

This preparation path resamples the selected HOI contact results and builds matching per-clip scaled object models. It uses the helper scripts in the same directory. Existing output metadata/checksums are checked for resumption. A 50 Hz file is a resampled reference, not evidence that the optimizer ran at 50 Hz or that training is stable.

Audit trusted local contact pickles:

```bash
python retargeting/lift/scripts/audit_lift_contact.py \
  --batch ~/holosoma/t2/hoi_all_lift_candidates01 \
  --out ~/holosoma/t2/hoi_all_lift_candidates01/contact_mode_audit.json
```

Only load your own generated pickles. The current repository does not yet contain a returned per-clip contact audit of all 24 originals. The batch source's explicit command, the single-clip log and the uploaded export inventory are separate kinds of evidence.

## Collect and upload the actual laptop files

The coding environment cannot read `/home/rudra` on the laptop. The uploader is the bridge. It reads the **passed selection**, requires `completed` statuses and the matching 50 Hz export records, validates mesh paths and checks available NPZ hashes before copying.

First update or clone this repository on the laptop. Git LFS is required for publishing motion arrays, pickles, videos and mesh assets. Authenticate Git using your normal GitHub setup; no token belongs in this script or repository.

```bash
# Run outside an existing clone; skip this clone command if you already have one.
git clone https://github.com/SuryanshSingh06/ForceLocoManipulation.git
cd ForceLocoManipulation

# Collect locally without committing or uploading.
python3 retargeting/lift/scripts/upload_lift_assets.py \
  --holosoma-root ~/holosoma \
  --expected-count 24
```

Inspect `retargeting/lift/artifacts/selected24/upload_inventory.json`. Then publish from an isolated worktree using a new branch:

```bash
python3 retargeting/lift/scripts/upload_lift_assets.py \
  --holosoma-root ~/holosoma \
  --expected-count 24 \
  --push
```

If the rendered videos are in Downloads rather than the clip folders, add:

```bash
--video-dir ~/Downloads/retargeted_lift24_videos
```

`--push` does not reuse the local collection directory. It starts from the current committed handoff checkout, collects afresh in a clean worktree, commits only that dataset directory and pushes a new `lift-retargeting-assets-...` branch. It does not force-push or automatically merge into `main`. It refuses unpublished base changes outside the lift handoff. Share the printed branch URL with the reviewer; a later merge can bring that payload onto `main`.

The payload includes selected contact/kinematic results, optional resampled pickles and videos, statuses, selections, logs, quality plots/curves, NPZ exports and per-clip model dependencies. It also snapshots available local helper scripts because the laptop versions may differ from these recovered copies. Original `.pt` datasets, SMPL/SMPL-X body models, checkpoints and other action classes are excluded. Review third-party asset redistribution terms before adding external data beyond this scope.

If the expected count changes intentionally, use a new selection and dataset name, for example `--expected-count 30 --dataset-name selected30`. Do not silently truncate a 69-folder directory to 24 clips.

## Remaining work

- Compare the intended physical rigid hand with the exact robot assets used for reference generation and training.
- Use one consistent robot/hand frame and collider setup across HOI, export and simulation.
- Correct contact targets and rerun one clip before rerunning the selected batch.
- Verify hand-surface contact across the raised interval and inspect penetration/sliding.
- Validate physical support with dynamic control and a free object. Kinematic playback is insufficient.
- Fill calibrated palm offsets/object collider dimensions before relying on the offline QA tool.

The half-sphere-hand experiment is documented as a diagnostic mismatch, not recommended as the final robot. A rubber-hand preview script was prepared earlier but no completed live test or HOI hand-model patch was supplied. A later stated preference was to keep the standard G1. No Newton migration or final hand-model replacement is claimed by this handoff.
