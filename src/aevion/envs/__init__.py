from .domain_rand import DomainRandomization
from .swarm_env import SWARM_TASKS, ParallelSwarmEnv, SwarmEnvConfig, VecSwarmEnv, make_swarm_sb3_vec_env, parallel_env
from .tasks import ConnectedNavigationTask, HoverTask, NavigationTask, TrackingTask, WaypointTask, make_task
from .tasks_ext import AreaCoverageTask, GateRacingTask, InspectionTask, LandingTask, TargetTrackingTask
from .vec_env import EnvConfig, VecQuadEnv

try:
    from .gym_env import QuadrotorEnv, QuadVectorEnv, make_sb3_vec_env
except Exception:  # gymnasium missing
    QuadrotorEnv = None  # type: ignore[assignment,misc]
    QuadVectorEnv = None  # type: ignore[assignment,misc]
    make_sb3_vec_env = None  # type: ignore[assignment]

__all__ = [
    "DomainRandomization",
    "EnvConfig",
    "VecQuadEnv",
    "QuadrotorEnv",
    "QuadVectorEnv",
    "make_sb3_vec_env",
    "HoverTask",
    "WaypointTask",
    "TrackingTask",
    "NavigationTask",
    "ConnectedNavigationTask",
    "GateRacingTask",
    "LandingTask",
    "TargetTrackingTask",
    "InspectionTask",
    "AreaCoverageTask",
    "make_task",
    "SwarmEnvConfig",
    "VecSwarmEnv",
    "ParallelSwarmEnv",
    "parallel_env",
    "make_swarm_sb3_vec_env",
    "SWARM_TASKS",
]
