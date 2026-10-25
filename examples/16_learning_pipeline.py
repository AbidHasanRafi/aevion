"""A learning pipeline on the landing task: scripted expert -> DAgger -> ONNX export, with a curriculum and a tracked log.

python examples/16_learning_pipeline.py --out learning_pipeline
"""

import argparse
from pathlib import Path

import numpy as np

from aevion.envs import VecQuadEnv
from aevion.envs.experts import make_expert
from aevion.ml import Curriculum, JsonlLogger, collect, dagger, evaluate_policy_rollouts, export_onnx
from aevion.ml.policies import verify_onnx


def landing_env(seed: int, platform_speed: float = 0.8) -> VecQuadEnv:
    env = VecQuadEnv(16, task="landing", action_mode="velocity", seed=seed, max_episode_s=14.0, control_dt=0.05, physics_dt=0.01)
    env.task.platform_speed = (0.0, platform_speed)
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="learning_pipeline")
    ap.add_argument("--iterations", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. a curriculum: the platform starts still and speeds up as the scripted expert masters each level
    env = landing_env(0)
    expert = make_expert(env)
    cur = Curriculum([0.0, 0.5, 1.0], promote_at=0.9, window=16)
    cur.apply(env)
    obs = env.reset(seed=1)
    for _ in range(env.max_steps * 2):
        obs, r, te, tr, info = env.step(expert(obs))
        done = np.asarray(env.B.to_numpy(info["done"])).astype(bool)
        if done.any() and cur.report(np.asarray(env.B.to_numpy(info["success"]))[done]):
            cur.apply(env)
    print(
        f"curriculum: level {cur.level} of {len(cur.stages) - 1} reached after {cur.episodes} episodes (platform speed up to {env.task.platform_speed[1]:.2f} m/s)"
    )

    # 2. demonstrations, then DAgger: cloning alone versus aggregating the learner's own states
    env = landing_env(0)
    expert = make_expert(env)
    demos = collect(env, expert, 250, seed=1)
    demos.save(out / "demonstrations.npz")
    print(f"dataset   : {len(demos)} transitions, {len(demos.returns())} finished episodes, saved as D4RL-style npz")
    with JsonlLogger(out / "dagger.jsonl") as log:
        result = dagger(
            env, expert, iterations=args.iterations, steps_per_iter=250, hidden=(64, 64), epochs=40, eval_env=landing_env(5), eval_episodes=32
        )
        for row in result.history:
            log.log(row["iteration"], **{k: v for k, v in row.items() if k != "iteration"})
            print(
                f"DAgger {row['iteration']}  : dataset {int(row['dataset']):>6}, loss {row['loss']:.2e}, success {row['success_rate']:.2f}, crashes {row['crash_rate']:.2f}"
            )

    # 3. export and check the exported graph
    path = export_onnx(result.policy, out / "landing_policy.onnx")
    try:
        err = verify_onnx(result.policy, path)
        print(f"export    : {path.name}, ONNX Runtime output differs from PyTorch by {err:.1e}")
    except ImportError:
        print(f"export    : {path.name} (onnxruntime is not installed, so the graph was not verified)")
    fresh = landing_env(9)
    stats = evaluate_policy_rollouts(fresh, lambda o: fresh.B.asarray(result.policy(fresh.B.to_numpy(o))), 32, seed=3)
    print(f"held-out  : success {stats['success_rate']:.2f}, crash {stats['crash_rate']:.2f} over {stats['episodes']} episodes on new seeds")


if __name__ == "__main__":
    main()
