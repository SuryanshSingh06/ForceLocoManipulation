# Holosoma lift recovery, version 2

Run in hsretargeting. Defaults read the 104 attempts in
~/holosoma/t2/holosoma_lift_all03, preserve its completed outputs (88 in the
reported run), and retry its 16 failures into holosoma_lift_all04. Successful
matching selected IDs are copied into holosoma_lift_selected24_04. The old
outputs, source recordings and installed solver are not edited.

The process-local solve wrapper first uses original settings. Infeasible steps
are retried at trust-region radii 0.4 and 0.8, then foot-sticking XY tolerances
3, 5 and 10 mm if needed. Collision and joint-limit flags stay enabled.
Per-frame foot tolerances can accumulate drift; larger updates may cause jumps.

If a frame solve remains infeasible, version 2 follows its object path using
4, 8, 16, then 32 internal subdivisions. Translation is linear; orientation uses
SLERP. Each attempt restarts from the previous output pose. The current frame's
interaction target stays in effect. Foot anchoring stays at the previous OUTPUT
frame across internal steps. Internal poses are discarded, so frame count and
requested object endpoints are unchanged. This is a numerical recovery device,
not a physical support simulation. Blocked paths or incompatible constraints
may still fail. A failed recovery does not prove the real robot cannot do it.

Newly solved poses are checked with the installed retargeter's nonlinear
collision-distance queries and joint limits. Collision residual may exceed the
configured penetration tolerance by at most 0.2 mm; joint-limit excess is capped
at 1e-5. Checks cover the masks and pairs queried by this retargeter, plus any
enabled self-collision pairs, not every possible physical collision. Every
native output frame of a newly solved clip is audited before completion.
Existing successful outputs are hash checked and preserved, NOT retrospectively
certified by the new audit. All outputs need motion and grasp-quality review.

Per-clip records include retarget.log, recovery_events.json, constraint_audit.json
(for new solves), selection.json, status.json, motion_original.npz and
motion_fps50.npz. Preserved results retain their old status in
preserved_status.json. Remaining failures are listed in failures.json. Failed
frames are never frozen or marked completed. Qpos playback NPZ is not the final
Holosoma training-array schema or proof of physical hand support.

Run and resume:

    python ~/holosoma/t2/holosoma_lift_recovery/t2_recover_holosoma_lifts.py

The same command resumes using hash/settings checks. This version uses new 04
output folders. Required dependencies and demo_data/height_dict.pkl are checked;
private runtime models are copied from the previous run.

Validation: six isolated tests and syntax checks passed. Tests cover unchanged
successful solves, bounded retries, restoration of settings, exhaustion remains
failed, unrelated errors not retried, exact object endpoints, fixed previous-frame
foot anchors, and rejection of nonlinear penetration. Full optimization on the
laptop recordings cannot be run in this authoring environment. Success for all
104 is unverified.

Upstream solver inspected:
https://github.com/amazon-far/holosoma/blob/main/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py
