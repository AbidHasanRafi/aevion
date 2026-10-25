"""Multi-agent coordination: communication graphs, consensus and formation control, task allocation, coverage, UTM-style deconfliction.

``swarm`` (lossy inter-agent link and collision-avoidance offsets) and ``formations`` are the basic pieces; the rest is the algorithm library.
"""

from .allocation import CBBAResult, TravelScore, auction, brute_force_allocation, cbba, hungarian
from .consensus import average_consensus, consensus_rate, formation_error, formation_step, lossy_consensus, push_sum
from .coverage import CoverageDomain, LloydResult, centroids, coverage_cost, lloyd, lloyd_step, voronoi_assign
from .formations import FORMATIONS, make_formation
from .graphs import adjacency, algebraic_connectivity, disk_graph, knn_graph, laplacian, metropolis_weights
from .laws import CoverageLaw, FormationKeeping, Rendezvous, SwarmLaw
from .swarm import CommModel, SwarmCoordinator, min_separation
from .utm import SpaceTimePlanner, Trajectory4D, UTMPlan, detect_conflicts, predict_conflicts, schedule_departures

__all__ = [
    "CommModel",
    "SwarmCoordinator",
    "min_separation",
    "FORMATIONS",
    "make_formation",
    "adjacency",
    "algebraic_connectivity",
    "disk_graph",
    "knn_graph",
    "laplacian",
    "metropolis_weights",
    "average_consensus",
    "consensus_rate",
    "formation_error",
    "formation_step",
    "lossy_consensus",
    "push_sum",
    "hungarian",
    "auction",
    "cbba",
    "CBBAResult",
    "TravelScore",
    "brute_force_allocation",
    "CoverageDomain",
    "LloydResult",
    "centroids",
    "coverage_cost",
    "lloyd",
    "lloyd_step",
    "voronoi_assign",
    "SwarmLaw",
    "FormationKeeping",
    "Rendezvous",
    "CoverageLaw",
    "Trajectory4D",
    "SpaceTimePlanner",
    "UTMPlan",
    "detect_conflicts",
    "predict_conflicts",
    "schedule_departures",
]
