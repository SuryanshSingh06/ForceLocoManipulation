#!/usr/bin/env python3
"""Continue the exact saved large-box experiment; preserve weights and optimizers."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import importlib
import json
import subprocess
import sys
import traceback
import zipfile

DEFAULT = '/home/rudra/holosoma/t3/largebox_fresh_20261011_044737_022271/logs/LargeBoxFresh/20261011_044744-sub7_largebox_024_t1_lift_000009_000066-locomotion/model_00999.pt'


def worker(args):
    sys.path.insert(0, str(args.out))
    base = importlib.import_module('run_largebox_fresh')
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.config_types.logger import DisabledLoggerConfig
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    if saved.training.num_envs != 128:
        raise RuntimeError('This continuation requires the original 128 environments')
    cfg = replace(saved,
        training=replace(saved.training, checkpoint=str(args.checkpoint), headless=True, export_onnx=False),
        logger=DisabledLoggerConfig(base_dir=str(args.out/'logs')),
        algo=replace(saved.algo, config=replace(saved.algo.config,
            num_learning_iterations=args.iterations, load_optimizer=True)))
    cfg.save_config(str(args.out/'resume_config.yaml'))
    original_build = base.build_config
    original_check = base.check_reward_math
    restores = []
    def saved_config(*unused):
        return cfg
    def check_and_install(device):
        # base.worker calls this only after TrainingContext launches IsaacSim.
        from holosoma.agents.ppo.ppo import PPO
        original_load = PPO.load
        def checked_load(algo, path):
            if path is None:
                raise RuntimeError('Continuation must load a checkpoint')
            result = original_load(algo, path)
            iteration = int(algo.current_learning_iteration)
            algo.current_learning_iteration = iteration + 1
            base.write_json(args.out/'checkpoint_load.json', dict(
                checkpoint=str(path), loaded_iteration=iteration, first_update=iteration+1,
                actor_and_critic_loaded=True, optimizer_loaded=True,
                normalization_enabled=algo.empirical_normalization,
                environment_restore_called=True))
            return result
        PPO.load = checked_load
        restores.append((PPO, original_load))
        return original_check(device)
    base.build_config = saved_config
    base.check_reward_math = check_and_install
    try:
        base.worker(args)
    finally:
        base.build_config = original_build
        base.check_reward_math = original_check
        for cls, original in restores:
            cls.load = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(DEFAULT))
    parser.add_argument('--iterations', type=int, default=2000, help='Additional updates, not total')
    parser.add_argument('--repo', type=Path, default=Path.home()/'holosoma')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.num_envs = 128
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.worker:
        try:
            worker(args)
        except BaseException:
            (args.out/'worker_exception.txt').write_text(traceback.format_exc())
            raise
        return
    if not args.checkpoint.is_file():
        raise SystemExit('Checkpoint missing: '+str(args.checkpoint))
    source = next((p for p in args.checkpoint.parents if (p/'run_largebox_fresh.py').is_file()), None)
    if source is None:
        raise SystemExit('Checkpoint must belong to the new large-box experiment')
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.repo, text=True).strip()
    if revision != 'd18d6cc50f872c15e904a22ceac22313cec955c8':
        raise SystemExit('Unexpected Holosoma revision: '+revision)
    inputs = json.loads((source/'pilot_inputs.json').read_text())
    if inputs.get('hold_starts') != [92,79] or len(inputs.get('clip_ids',[])) != 2:
        raise SystemExit('Unexpected source clip layout')
    for key in ('motion','object_urdf'):
        if not Path(inputs[key]).is_file():
            raise SystemExit('Original prepared asset missing: '+inputs[key])
    if args.out is None:
        args.out = args.repo/'t3'/('largebox_continue_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for filename in ('run_largebox_fresh.py','pilot_inputs.json'):
        (args.out/filename).write_bytes((source/filename).read_bytes())
    (args.out/'resume_largebox.py').write_bytes(Path(__file__).read_bytes())
    command = [sys.executable,'-u',str(args.out/'resume_largebox.py'),'--worker',
        '--repo',str(args.repo),'--checkpoint',str(args.checkpoint),
        '--out',str(args.out),'--iterations',str(args.iterations)]
    code = 1
    with (args.out/'console.log').open('w') as log:
        process = subprocess.Popen(command,cwd=args.repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line);log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            code = process.wait()
    finite_path = args.out/'finite_rollout_check.json'
    finite = json.loads(finite_path.read_text()) if finite_path.exists() else {}
    checkpoints = sorted(args.out.glob('logs/**/model_*.pt'))
    completed = code == 0 and not (args.out/'worker_exception.txt').exists() and finite.get('updates_checked') == args.iterations and bool(checkpoints) and (args.out/'checkpoint_load.json').exists()
    (args.out/'result.json').write_text(json.dumps(dict(training_completed=completed,
        returncode=code,additional_updates=args.iterations,
        updates_completed=finite.get('updates_checked',0),source_checkpoint=str(args.checkpoint),
        checkpoints=[str(p) for p in checkpoints],physical_success_verified=False),indent=2))
    report = args.out/'largebox_continue_report.zip'
    with zipfile.ZipFile(report,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(args.out.rglob('*')):
            if file.is_file() and file.suffix in {'.json','.jsonl','.yaml','.log','.txt','.py'}:
                archive.write(file,file.relative_to(args.out))
    print('\nREPORT:',report,flush=True)
    if checkpoints:
        print('LATEST CHECKPOINT:',checkpoints[-1],flush=True)
    if not completed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
