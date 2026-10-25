"""Train PPO (Stable-Baselines3) on a randomised hover task, then evaluate with confidence intervals.

pip install stable-baselines3
"""

from stable_baselines3 import PPO

from aevion.envs import DomainRandomization, VecQuadEnv, make_sb3_vec_env
from aevion.experiments import ProportionalPolicy, evaluate_policy

kw = dict(task="hover", action_mode="velocity", max_episode_s=5.0, domain_randomization=DomainRandomization.mild())
env = make_sb3_vec_env(64, **kw)  # device="cuda" below if you have a GPU
model = PPO("MlpPolicy", env, n_steps=128, batch_size=1024, learning_rate=3e-4, device="cpu", verbose=1)
model.learn(total_timesteps=400_000)


def mk(seed):  # a fresh environment per evaluation seed
    return VecQuadEnv(32, seed=seed, **kw)


learned = evaluate_policy(lambda obs: model.predict(obs, deterministic=True)[0], mk, seeds=range(5), episodes_per_seed=32)
baseline = evaluate_policy(ProportionalPolicy(mk(0)), mk, seeds=range(5), episodes_per_seed=32)
print("learned policy\n" + learned.text())
print("proportional baseline\n" + baseline.text())
