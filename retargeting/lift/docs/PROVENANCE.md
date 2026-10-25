# Scope and provenance

Prepared for review on 2026-10-25. This folder contains recovered script snapshots, supplied diagnostic files and a newly written uploader/auditor. It does not claim to be a checkout of the exact installed Holosoma or HOI revisions.

## What is included now

- The recovered lift-only `t2_batch_retarget.py` snapshot, which explicitly generates HOI contact output and quality-screens it.
- The export/resampling/geometry helper snapshots recovered from `contact_qa_tools.zip`.
- The newer paused viewer/diagnostic helper snapshots recovered from `contact_scene_debug.zip`.
- The supplied two Isaac pose snapshots and stage collider inventory.
- An unfilled calibration template, explicitly not a completed calibration.
- A compact lift24 inventory extracted from the supplied export inventory.
- New upload and contact-audit utilities, plus a Bash wrapper and documentation.

Recovered helpers may differ from scripts modified locally on the laptop. The uploader includes available local versions in `local_scripts/` so a reviewer can compare them. The older standalone `t2_export_holosoma.py` CLI defaults reflect the historical mixed 32-clip export; invoke it through the lift-only preparation driver rather than using that standalone default for this dataset.

No full source motion datasets, body-model weights, policy checkpoints, full third-party repositories or unrelated action outputs are added by this change. The earlier repository files remain untouched except for the root README navigation update.

## What has not been uploaded or verified here

- Actual 24 contact pickles, all source/retargeted plots and current generation logs from the laptop.
- The complete current passed selection and NPZ export manifest from that laptop.
- A returned per-clip contact-mode audit of all 24 originals.
- A completed replacement of the HOI hand model or retargeting surface-contact objective.
- Physical support forces or a dynamic success rate for these references.
- A standard-G1/HOI/Isaac/Newton asset configuration validated as a single consistent pipeline.

The uploader collects the selected local artifacts and creates a checksum inventory before Git publication. It pushes on a fresh branch, preserving other work. A mode audit and a geometry/physical QA report answer different questions and should be reviewed separately.

## Review questions for Francisco

1. Is the observed `--robot unitree_g1 --mode contact` HOI pipeline the intended method, or should this project use Holosoma's `--task-type object_interaction` route instead?
2. Do the HOI `palm_pad` contact frames match the intended rigid-hand surfaces and wrist offsets?
3. Are the active-link/contact windows appropriate for a two-handed lift, particularly the clip with no left-hand activation during flight?
4. What object collision approximation should be shared between generation and training, especially for the open plastic crate?
5. Should hand-surface proximity/nonpenetration be optimized directly in addition to point-marker tracking?
6. What dynamic grasp validation and reference acceptance criteria are required before AMP/training?

## Asset ownership and reproduction

Third-party robot/object assets remain subject to their upstream terms. SMPL-family body-model files are not part of this handoff. The local installation paths and environment names in examples are Rudra's recorded paths; adapt them explicitly rather than assuming they exist on another machine. Do not load pickle files from untrusted sources.
