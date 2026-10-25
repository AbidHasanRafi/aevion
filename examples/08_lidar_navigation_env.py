"""LiDAR obstacle-navigation environment: inspect observations and roll out a random policy."""

import numpy as np

from aevion import World
from aevion.envs import VecQuadEnv

world = World.random_forest(30, half_extent=10, seed=1)
env = VecQuadEnv(
    16,
    task="navigation",
    world=world,
    lidar=True,
    lidar_channels=1,
    lidar_samples=24,
    action_mode="velocity",
    bounds=(-14, -12, 0, 14, 12, 8),
    max_episode_s=15.0,
)
obs = env.reset(seed=0)
print("observation dim:", env.obs_dim, "= pose/velocity/attitude/action (22) + 24 LiDAR beams")
rng = np.random.default_rng(0)
crashes = 0
for _ in range(300):
    obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, (16, env.act_dim)))
    crashes += int(info["crashed"].sum())
print(f"random policy: {crashes} crashes in 300 steps × 16 environments")
