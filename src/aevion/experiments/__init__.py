"""Experiments: grids of trials with statistics (``Experiment``) and the toolkit around it - designs, sensitivity, Bayesian optimisation, caching, runners, database, suites."""

from .bayesopt import BOResult, GaussianProcess, bayes_opt
from .cache import ResultCache, canonical_json, trial_key
from .config_experiment import ConfigExperiment, Trial, deep_merge, nest
from .database import ResultDB, format_leaderboard, leaderboard
from .design import ParameterSpace, discrepancy, full_factorial, halton, latin_hypercube, sobol
from .engine import Experiment, ExperimentResult, apply_randomization
from .evaluate import EvalResult, ProportionalPolicy, TrainingLog, evaluate_policy
from .runners import ProcessRunner, Runner, SerialRunner, shard_indices, slurm_script
from .sensitivity import MorrisResult, SobolResult, morris, morris_analyze, morris_design, saltelli_design, sobol_analyze, sobol_indices
from .suites import SUITES, BenchmarkSuite, SuiteResult, get_suite, run_suite

__all__ = [
    "Experiment",
    "ExperimentResult",
    "apply_randomization",
    "EvalResult",
    "ProportionalPolicy",
    "TrainingLog",
    "evaluate_policy",
    "ConfigExperiment",
    "Trial",
    "deep_merge",
    "nest",
    "ResultCache",
    "canonical_json",
    "trial_key",
    "Runner",
    "SerialRunner",
    "ProcessRunner",
    "shard_indices",
    "slurm_script",
    "ResultDB",
    "leaderboard",
    "format_leaderboard",
    "BenchmarkSuite",
    "SuiteResult",
    "SUITES",
    "get_suite",
    "run_suite",
    "ParameterSpace",
    "latin_hypercube",
    "sobol",
    "halton",
    "full_factorial",
    "discrepancy",
    "morris",
    "morris_design",
    "morris_analyze",
    "MorrisResult",
    "saltelli_design",
    "sobol_analyze",
    "sobol_indices",
    "SobolResult",
    "GaussianProcess",
    "bayes_opt",
    "BOResult",
]
