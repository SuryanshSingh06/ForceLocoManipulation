# Holosoma G1 lift references: 102 completed clips

This folder contains **Holosoma retargeting outputs**, not HOI-Retarget outputs.
The task was object_interaction, robot g1, data format smplh. Of 104 lift
attempts, 102 completed and two failed. All 24 previously selected lift IDs are
included in the 102. No carry or push/pull motions are added by this upload.

| Object | Completed clips |
| --- | ---: |
| largebox | 41 |
| plasticbox | 35 |
| smallbox | 26 |

## File structure

```text
manifest.json                         Portable index of all 102 completed clips
clips/<clip-id>/motion_original.npz    Native frame-rate robot/object qpos
clips/<clip-id>/motion_fps50.npz       50 Hz quaternion-aware resampling
clips/<clip-id>/status.json            Original execution/provenance details
clips/<clip-id>/selection.json         T1 source recording and crop interval
clips/<clip-id>/retarget.log           Optimization log
clips/<clip-id>/recovery_events.json   Recovery events, when available
clips/<clip-id>/constraint_audit.json  Nonlinear audit, when available
selected24/manifest.json              References to the matching 24 shared clips
models/g1/                            Matching standard G1 model/geometry
models/<object>/                      Object URDF/mesh used by the runtime
failures/                             Two failed attempts: logs and metadata only
provenance/                           Original run/selection manifests and settings
scripts/                              Runner, recovery and installed Python snapshots
environment.txt                       Exporting environment package versions
checksums.json                        SHA256 and size of published files
```

Paths in manifest.json are relative to this folder. Paths in selected24/manifest.json
are relative to selected24/. Original status/provenance files retain historical
laptop paths as evidence; those are not portable loader paths.

## Motion format and use

qpos has 43 columns: root XYZ [0:3], root quaternion WXYZ [3:7], 29 G1 joint
positions [7:36], object XYZ [36:39], object quaternion WXYZ [39:43]. Read fps
from each NPZ. All motion_fps50 files have fps=50. Quaternion-aware interpolation
uses a uniform 50 Hz grid without extrapolating beyond the native final timestamp.
These are **qpos playback NPZs, not Holosoma joint_pos/body_pos training-schema
exports**. Training-array conversion and physical validation remain separate work.
Source human trajectories may also be present inside the NPZs. Original source
.pt recordings, licensed body-model weights, checkpoints and videos are not
added by this uploader. This is not a complete standalone source-data archive.
Preserve the recorded source filenames, checksums and [start:end] frame intervals
when obtaining source recordings to reproduce retargeting.

Robot/object geometry is copied from the private runtime without deliberate
rescaling. Mesh references in copied XML/URDF files are made relative; original
installed models are not edited. Some combined object MJCFs were generated from
the largebox template and retain its inertial parameters: these are kinematic
reference assets, not calibrated physical mass/inertia models.

## Generation and recovery

The native runner used --task-type object_interaction --robot g1 --data-format
smplh with object nonpenetration, joint limits and foot sticking enabled.
Recovery first enlarged the pose-update trust region to at most 0.8, then
allowed foot-sticking XY tolerances up to 10 mm if needed. The final recovery
also tried 4/8/16/32 internal object-motion subdivisions, preserving each output
frame's requested object endpoint and retaining previous-output-frame foot
anchoring. See script snapshots and per-clip events for exact behavior.

The final run preserved 88 earlier results and newly recovered 14. Only newly
solved clips received that version's frame-by-frame nonlinear constraint audit.
Absence of an audit is not a pass. Checks cover the retargeter's configured
collision masks/pairs and joint limits; they are not physical support tests.
Per-frame foot tolerances can accumulate drift. Larger updates can cause jumps.
Every clip remains pending robot-motion review, regardless of its T1 review.
Completion does not certify hand contact, realistic grasp forces, stability,
collision-free training behavior, or suitability as an AMP reference.

## The two failures

- sub1_plasticbox_046_t1_lift_000024_000107
- sub3_largebox_035_t1_lift_000026_000081

They have logs/metadata under failures/ and are excluded from the motion index.

## Download

Clone this published branch with Git LFS, then run git lfs pull and git lfs fsck.
checksums.json can verify the hydrated asset bytes. Existing HOI selected-24
artifacts remain elsewhere in this branch under retargeting/lift/artifacts/selected24;
they use a different backend and are not replaced by these Holosoma results.
The model/software licensing terms of the upstream projects still apply.
