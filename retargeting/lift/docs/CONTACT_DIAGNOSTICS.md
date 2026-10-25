# Hand contact: findings and open work

## What was measured

The detailed geometry investigation used the lift:

`sub12_plasticbox_036_t1_lift_000242_000280`

Uploaded Isaac snapshots at 50 Hz frames 29 and 51 gave the robot body poses and object pose in world coordinates, with WXYZ quaternions. The object's origin rose from approximately 0.61324 m to 0.87664 m, a change of 0.26340 m. This alone is not proof the object was physically supported because the viewer imposes the recorded poses.

The uploaded URDF and asset inventory confirmed:

- Robot: `g1/main_mesh_collision_halfspherehand.urdf` in that inspection session.
- Hand links: `left_sphere_hand_link`, `right_sphere_hand_link`.
- Hand mesh: `meshes/half_sphere.obj`, with collision approximation `convexHull`.
- Object: plastic-box mesh scaled by 0.82999998331, with a convex-hull collision approximation.
- Both hands, the box, wrists and elbows had collision enabled. Upper-arm capsule colliders were also enabled.

The hand collision origin in its link is zero. The fixed hand attachment in that URDF is approximately `xyz=(0.029, -0.003, 0)`, `rpy=(0, 1.57, 0)` on each wrist. The standard rubber-hand donor specifies different attachment transforms, including `x=0.0415` and zero RPY, with side-specific Y. Substituting only the visual mesh is therefore insufficient.

## Reconstructed convex-hull distances

| Saved frame | Left hand minimum gap | Right hand minimum gap |
| --- | ---: | ---: |
| 29 | 2.05 cm | 0.81 cm |
| 51 | 1.30 cm | 0.53 cm |

Method: construct convex hulls from the supplied meshes, apply the object mesh scale, transform hand vertices from the saved hand-link pose into the object's frame, and solve the minimum-distance problem between the two convex polyhedra. The measured distances were checked against separating-axis support bounds; solver feasibility residuals were near numerical precision.

These are nominal reconstructed hull distances. The run did not retrieve PhysX's cooked hull vertices, rest/contact offsets, material response or live filtered contact forces. Do not describe these values as measured PhysX forces, penetration depths or a dynamic grasp success rate. Some debug USD world transforms in the stage inventory were stale under Fabric; the calculation used the explicit saved articulation/object pose arrays instead.

## Why optimizer marker residuals were misleading

The earlier batch diagnostic showed most active `left_palm_pad` and `right_palm_pad` target errors around 0.1 to 0.4 cm median. Those contact markers belong to the HOI robot representation, whereas the Isaac inspection used half-sphere hands. Names, origins and surface locations must be reconciled.

Two notable earlier outputs:

- `sub10_largebox_033_t1_lift_000010_000043`: left palm inactive in all 32 reported airborne frames, while the right was active throughout.
- `sub7_largebox_029_t1_lift_000010_000090`: left maximum marker error 7.4 cm; right maximum 6.0 cm, with medians 1.6 cm and 0.9 cm respectively.

These values are transcribed session output. The original all-frame curves and contact pickles remain on the laptop and should be uploaded for reproduction. Do not generalize two mesh snapshots or these marker summaries into acceptance of every frame in all 24 clips.

## Rubber-hand comparison

The uploaded standard G1 URDF has rigid rubber-hand visual links, with 0.170 kg hand masses and donor inertias. Their STL bounding boxes matched the reference export's rubber-hand OBJ bounding boxes in the local comparison. However, the donor rubber-hand links had no `<collision>` entries.

A preview helper was prepared to build an isolated candidate with explicit rubber-hand mesh colliders, donor inertias and attachment transforms, while preserving the 29 actuated joints and legacy hand link names for config compatibility. It was locally checked against fixtures. No successful live run of that candidate or completed corresponding HOI patch was returned. It is not installed as the default by this repository.

The same two saved joint poses evaluated with the nominal rubber-hand geometry still gave either separation or overlap depending on side/frame. Changing a mesh does not repair a trajectory.

The user's subsequent direction was to keep the standard G1 and address the hand mismatch separately. Standard G1 identity alone is not a complete specification of a physical hand attachment. No final physical attachment geometry match was supplied.

## Collider visualization and viewer fixes

The viewer initially failed to import the box because a hashed mesh basename started with a digit and the URDF importer attempted to use it as a USD identifier. The helper now creates debug-only `mesh_...` aliases, verifies mesh bytes, preserves scale/origin, and leaves originals intact.

A second failure was `Set 'has_object' to true, but loaded no rigid bodies in the scene.` The debug object had been configured `fixed=True`, making it static in the object's registry. The helper now uses `fixed=False` for registration and imposes poses without stepping physics. This is a geometry-viewer configuration, not a training or dynamic hold test.

In the shown Isaac GUI, the viewport sliders menu exposed **Show By Type → Physics → Colliders → All**. Earlier advice to use a `Window → Simulation` menu did not match that application layout. Even after enabling the overlay, some shapes were not drawn. The uploaded stage inventory established enabled collision APIs; missing outlines alone were not evidence of disabled collision.

## Required correction and validation

1. Choose and verify the intended rigid-hand geometry in the robot asset, including wrist transform and collision approximation.
2. Use matching kinematics/contact surfaces in HOI and the simulation/export model.
3. Place targets on appropriate object/hand surfaces; check active-link gating during the airborne interval.
4. Re-solve one clip with smoothness, joint limits and penetration checks. Increasing a marker weight alone is not a geometry correction.
5. Recompute body poses and velocities when exporting; resample at 50 Hz consistently.
6. Measure both hand-surface gaps over the full raised interval, then inspect visually.
7. Test dynamic physical support with a free object under control. Never infer support forces from imposed reference playback.

A rectangular box or union of boxes can be a useful collision approximation only after its dimensions, placement and intended hand surfaces are measured. A single convex hull fills the open interior/holes of the crate. Altering an approximation changes the interaction and must be tracked in both reference and simulation assets.

## Calibration status

`evidence/calibration.json` is an unfilled template: robot geometry is unverified, palm offsets are null, and object half extents are null. The offline geometry QA thresholds in it are project heuristics, not universal grasp criteria. Do not mark this template verified without actual measurements.
