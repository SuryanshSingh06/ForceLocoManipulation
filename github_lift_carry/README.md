# G1 lift and carry motion references

19 retained clips: 9 lift and 10 carry. All motions are sampled at 50 FPS.

## Contents
- motions/: Holosoma-layout NPZ motion references.
- models/: matching G1/object MJCFs, scaled object URDFs, and referenced assets.
- hoi_selected40/: intermediate corrected 50 FPS HOI pickles.
- previews/: available videos rendered from the exported NPZ files.
- scripts/: retargeting, resampling, and NPZ export scripts.
- manifest.json: retained clips, frame ranges, source checksums, model paths,
  object scale, and motion paths.
- selected40_manifest.json: original lift/carry selection, including rejection.
- retained_t1_manifest.json: T1 inputs corresponding to retained motions.
- t2_rejections.json: rejected lift/carry IDs.
- npz_schema_example.json: example array shapes and data types.
- software_provenance.json: repository commits and local modification status.
- environment_snapshot.txt: Python package snapshot.

## Format
Positions are in metres, joint angles in radians, and quaternions in WXYZ order.
World coordinates follow the MuJoCo Z-up models.
Read joint_names and body_names from each NPZ for ordering.
joint_pos contains the robot free base plus 29 joint positions.
joint_vel contains the robot base velocities plus 29 joint velocities.
Object pose and velocity are stored separately. qpos includes both robot and
object and can be used for MuJoCo replay with the matching model_xml.
Use the scaled model paths recorded in manifest.json.

## Loading
    import numpy as np
    with np.load("motions/lift/CLIP_ID.npz", allow_pickle=False) as motion:
        print(motion.files)
        print(motion["joint_pos"].shape)

## Reproduction
Install Holosoma and HOI-Retarget using the recorded software versions.
Obtain source recordings and required SMPL-X models separately.
Source recording paths and intermediate pickle metadata refer to the original
machine and must be adjusted when reproducing on another machine.

Retarget retained inputs:
    python scripts/t2_batch_retarget.py --manifest retained_t1_manifest.json --repo /path/to/hoi-retarget --out /path/to/new-output

Convert each corrected HOI motion:
    python scripts/t2_resample_hoi.py --input /path/to/contact_window.pkl --fps 50

Export to Holosoma layout using t2_export_holosoma.py and the corresponding
selection/rejection files and HOI output directory.

## Training status
These are motion references for T3 lift/carry training.
Training environments, rewards, physics settings, and policies are not included.
Object masses are template values, not measured masses.
Configure matching object geometry/scale and verify the training loader before use.
A successful export does not certify physical task execution.
External data and asset licenses continue to apply.
