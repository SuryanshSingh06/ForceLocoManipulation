# T3 carry policy

This task trains one Unitree G1 29-DoF object-aware whole-body-tracking policy
on the six accepted small-box carry references from T2. The two repositories
are expected to be sibling checkouts:

```text
parent/
├── ForceLocoManipulation/
└── holosoma/
```

The carry preset is based on HoloSoma commit
`d18d6cc50f872c15e904a22ceac22313cec955c8`. It uses:

- `t3/carry/motions/smallbox/` as a six-clip `MultiMotionLoader` dataset;
- the reviewed, scaled small-box URDF from T2;
- a 1.0 kg box mass;
- static and dynamic friction coefficients of 0.9;
- HoloSoma's existing PPO object-aware WBT observations and rewards.

## Apply the HoloSoma change

From a clean HoloSoma checkout:

```bash
git checkout d18d6cc50f872c15e904a22ceac22313cec955c8
git switch -c t3-carry
git am ../ForceLocoManipulation/t3/carry/patches/*.patch
```

If the repositories are not siblings, set the project root before launching:

```bash
export FORCE_LOCO_DATA_ROOT=/absolute/path/to/ForceLocoManipulation
```

## Validate the data

This check uses only the Python standard library and can run on macOS:

```bash
python t3/carry/scripts/validate_carry_data.py
```

## Linux/Isaac Sim smoke test

From the HoloSoma checkout:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
  exp:g1-29dof-carry \
  --training.headless=False \
  --training.num_envs=1
```

## Training

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-carry \
  simulator:isaacsim \
  logger:wandb \
  --training.seed=1 \
  --training.num-envs=4096
```

Begin with the replay and a short 64-environment training smoke test before
launching the full run. Isaac Sim training has not been executed on macOS.

