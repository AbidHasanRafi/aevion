"""Registers every built-in component. Imported once by :mod:`aevion.registry`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..control import advanced as A
from ..control import controllers as C
from ..control import reference as R
from ..engine import effects as EF
from ..engine import faults as F
from ..engine import hooks as H
from ..engine import sensor_faults as SF
from ..environments import wind as W
from ..environments import wind_fields as WF
from ..environments.actors import ACTORS
from ..environments.geofence import GeofenceCheck
from ..environments.world import World
from ..estimation import ESKFEstimator, Estimator
from ..health import checks as HC
from ..multiagent import formations as FM
from ..multiagent import laws as SL
from ..planning.integrate import mission_trajectory, planned_trajectory
from ..sensors import (
    GPS,
    IMU,
    Barometer,
    Camera,
    EventCamera,
    GnssReceiver,
    Lidar,
    Magnetometer,
    OpticalFlow,
    PitotTube,
    Radar,
    RangeFinder,
    Ultrasonic,
    UwbRanging,
)
from ..sensors.scan_patterns import PATTERNS as SCAN_PATTERNS
from ..vehicles.spec import PRESETS, VehicleSpec
from ..wireless import AirToGroundChannel, CellularNetwork
from ..wireless.antennas import PATTERNS as ANTENNA_PATTERNS
from ..wireless.mac_network import SCHEDULERS, MeshNetwork, NrCellularNetwork
from ..wireless.propagation import MODELS as PROPAGATION_MODELS
from ..wireless.security import GnssJamming, GnssSpoofing, NetworkJamming
from . import registries as _r  # noqa: F401  (circular-safe: registries object exists before builtin runs)

if TYPE_CHECKING:
    from ..engine.simulation import Simulation


def _register_all(r: Any) -> None:
    # ----------------------------------------------------------------- vehicles
    for name, fn in PRESETS.items():
        r.vehicles.register(name, fn, doc=(fn.__doc__ or "").strip().split("\n")[0])
    r.vehicles.register("quad_x", VehicleSpec.quad_x, doc="generic X-configuration quadrotor (set arm, mass, kT, ...)")
    r.vehicles.register("quad_plus", VehicleSpec.quad_plus, doc="generic +-configuration quadrotor")
    r.vehicles.register("hexa", VehicleSpec.hexa, doc="generic hexacopter")
    r.vehicles.register("octo", VehicleSpec.octo, doc="generic octocopter")

    # -------------------------------------------------------------- controllers
    def geometric(
        sim: "Simulation",
        wn_pos: Any = None,
        zeta_pos: Any = 0.95,
        wn_att: Any = None,
        zeta_att: Any = 0.9,
        ki_pos: Any = 0.0,
        int_limit: Any = 2.0,
        max_tilt_deg: Any = 55.0,
    ) -> Any:
        """Geometric SE(3) tracking controller (Lee et al. 2010)"""
        g = C.Gains(wn_pos=wn_pos, zeta_pos=zeta_pos, wn_att=wn_att, zeta_att=zeta_att, ki_pos=ki_pos, int_limit=int_limit, max_tilt_deg=max_tilt_deg)
        return C.GeometricController(sim, g)

    r.controllers.register("geometric", geometric, aliases=("geometric_se3", "se3"))
    r.controllers.register(
        "cascaded", C.CascadedPID, doc="cascaded PID (position, velocity, attitude, rate), PX4-style", aliases=("cascaded_pid", "pid")
    )
    r.controllers.register("lqr", C.LQRHover, doc="LQR about hover with acceleration feedforward", aliases=("lqr_hover",))

    def _advanced(cls: Any, doc: str) -> Any:
        def make(sim: "Simulation", **params: Any) -> Any:
            gains = {k: params.pop(k) for k in ("wn_pos", "zeta_pos", "wn_att", "zeta_att", "ki_pos", "int_limit", "max_tilt_deg") if k in params}
            return cls(sim, gains=C.Gains(**gains) if gains else None, **params)

        make.__doc__ = doc
        make.__name__ = cls.__name__
        return make

    r.controllers.register(
        "mpc", _advanced(A.MPCController, "constrained linear MPC with a Kalman disturbance estimator (offset-free)"), aliases=("linear_mpc",)
    )
    r.controllers.register("mppi", _advanced(A.MPPIController, "sampling-based model-predictive path-integral control (obstacle-aware)"))
    r.controllers.register("indi", _advanced(A.INDIController, "incremental nonlinear dynamic inversion torque law (inertia-model free)"))
    r.controllers.register("l1", _advanced(A.L1AdaptiveController, "L1 adaptive augmentation of the position loop"), aliases=("l1_adaptive",))
    r.controllers.register(
        "smc", _advanced(A.SlidingModeController, "sliding-mode translational loop with a boundary layer"), aliases=("sliding_mode",)
    )

    # ------------------------------------------------------------- trajectories
    def figure8(center: Any = (0, 0, 1.5), size: Any = 1.5, period: Any = 10.0) -> Any:
        """Lissajous figure-eight in the horizontal plane"""
        return R.Lissajous.figure8(center, size, period)

    r.trajectories.register("hover", R.Hover, aliases=("hold",))
    r.trajectories.register("circle", R.Circle)
    r.trajectories.register("figure8", figure8, aliases=("figure_8", "lemniscate"))
    r.trajectories.register("lissajous", R.Lissajous)
    r.trajectories.register("waypoints", R.Waypoints, doc="minimum-snap-style polynomial through waypoints", aliases=("min_snap",))
    r.trajectories.register("polyline", R.Polyline, doc="constant-speed straight segments")
    r.trajectories.register("planned", planned_trajectory, doc="collision-free minimum-snap trajectory planned around the world's obstacles")
    r.trajectories.register(
        "mission", mission_trajectory, doc="behaviour-tree mission (takeoff, goto, hover, land, conditions)", aliases=("behavior_tree",)
    )

    # ------------------------------------------------------------------ sensors
    r.sensors.register("imu", IMU, doc="6-axis MEMS IMU")
    r.sensors.register("gps", GPS, doc="GNSS receiver (Gauss-Markov error, latency)")
    r.sensors.register("gnss", GnssReceiver, doc="pseudorange GNSS receiver: constellations, atmosphere, multipath, NLoS, least-squares fix, DOP")
    r.sensors.register("pitot", PitotTube, doc="differential-pressure airspeed sensor (relative air velocity, lag, acceptance cone)")
    r.sensors.register("optical_flow", OpticalFlow, aliases=("flow",), doc="downward optical-flow sensor (raw and gyro-compensated flow, quality)")
    r.sensors.register("ultrasonic", Ultrasonic, doc="cone-beam ultrasonic rangefinder (nearest echo, oblique-surface loss, speed-of-sound bias)")
    r.sensors.register("uwb", UwbRanging, doc="UWB two-way ranging to fixed anchors (NLoS excess delay, outliers)")
    r.sensors.register("radar", Radar, doc="FMCW radar: radar-equation SNR, range/Doppler resolution, velocity ambiguity, optional range-Doppler map")
    r.sensors.register("baro", Barometer, aliases=("barometer",), doc="static-pressure altimeter")
    r.sensors.register("mag", Magnetometer, aliases=("magnetometer",), doc="3-axis magnetometer")
    r.sensors.register("rangefinder", RangeFinder, doc="single-beam range sensor")
    r.sensors.register(
        "lidar",
        Lidar,
        doc="spinning multi-beam LiDAR (config default: 8 x 180 beams, no point cloud)",
        defaults=dict(channels=8, samples=180, return_points=False),
    )
    r.sensors.register(
        "event_camera",
        EventCamera,
        aliases=("dvs",),
        doc="event camera: log-intensity change events with threshold mismatch, refractory period and background noise",
    )
    r.sensors.register("camera", Camera, doc="pinhole depth/RGB/segmentation camera (RL-sized 64 x 48 default)")

    # --------------------------------------------------------------- estimators
    r.estimators.register("mahony_kf", Estimator, doc="Mahony attitude filter + 9-state position/velocity Kalman filter", aliases=("default",))
    r.estimators.register(
        "eskf",
        ESKFEstimator,
        doc="15-state error-state Kalman filter (IMU + GNSS, baro, mag, range, flow, pose) with delayed fusion",
        aliases=("error_state_kf",),
    )

    # -------------------------------------------------------------------- winds
    def composite(models: Any = ()) -> Any:
        """sum of several wind models, given as specs"""
        return W.CompositeWind(*[r.winds.build(m) for m in models])

    r.winds.register("none", W.NoWind, aliases=("nowind", "calm"))
    r.winds.register("constant", W.ConstantWind, aliases=("constantwind",))
    r.winds.register("shear", W.WindShear, aliases=("windshear",), doc="atmospheric boundary-layer profile")
    r.winds.register("gust", W.Gust, doc="discrete 1-cos gust")
    r.winds.register("dryden", W.DrydenTurbulence, aliases=("drydenturbulence",), doc="Dryden turbulence (MIL-F-8785C)")
    r.winds.register("composite", composite, aliases=("compositewind",))
    r.winds.register(
        "log_law", WF.LogLawWind, aliases=("loglaw",), doc="neutral boundary-layer log profile with roughness length and displacement height"
    )
    r.winds.register("urban", WF.UrbanWind, doc="log-law wind modified around the world's buildings (cavity, wake, stagnation, downwash)")

    def gridded(file: Any = None, outside: Any = "nearest", scale: Any = 1.0, loop: Any = False) -> Any:
        """wind imported from a regular grid (.npz or .csv), trilinear in space, linear in time"""
        if file is None:
            raise TypeError("a gridded wind needs 'file' (an .npz or .csv path)")
        loader = WF.GriddedWind.from_csv if str(file).lower().endswith(".csv") else WF.GriddedWind.from_npz
        return loader(file, outside=outside, scale=scale, loop=loop)

    r.winds.register("gridded", gridded, aliases=("cfd",))
    r.winds.register(
        "random_constant", W.RandomConstantWind, aliases=("randomconstantwind",), doc="per-environment random constant wind (domain randomisation)"
    )

    # ------------------------------------------------------------------- worlds
    r.worlds.register("empty", lambda seed=0, **kw: World(**kw), doc="flat ground, no obstacles")
    r.worlds.register("forest", lambda seed=0, **kw: World.random_forest(seed=seed, **kw), doc="random tree trunks")
    r.worlds.register("city", lambda seed=0, **kw: World.city(seed=seed, **kw), doc="Manhattan-style building grid")
    r.worlds.register("corridor", lambda seed=0, **kw: World.corridor(seed=seed, **kw), doc="enclosed corridor with pillars")
    r.worlds.register("hills", lambda seed=0, **kw: World.hills(seed=seed, **kw), doc="smooth random terrain")
    from ..environments.scene import build_scene

    r.worlds.register(
        "scene",
        build_scene,
        aliases=("geo_scene", "mesh_scene"),
        doc="declarative scene: origin, weather, terrain files, OSM/GeoJSON buildings, meshes and primitives",
    )
    r.worlds.register("from_dict", lambda seed=0, data=None, **kw: World.from_dict(data), doc="a serialised world (World.to_dict())")

    # ----------------------------------------------------------------- channels
    r.channels.register("air_to_ground", AirToGroundChannel, doc="Al-Hourani air-to-ground channel (probabilistic or geometric LoS)")

    # ----------------------------------------------------------------- networks
    def cellular(world: Any = None, seed: int = 0, **d: Any) -> Any:
        """UAV as a cellular-connected user (base stations + air-to-ground channel)"""
        if "bs" not in d:
            raise TypeError("a cellular network needs 'bs' (a list of base stations)")
        return CellularNetwork.from_dict(d, world=world, seed=seed)

    r.networks.register("cellular", cellular)
    r.networks.register(
        "nr_cellular",
        NrCellularNetwork,
        doc="5G-NR-style cellular network: 3GPP path loss, antennas, fading, handover, RLF, scheduler, link adaptation",
    )
    r.networks.register("mesh", MeshNetwork, doc="UAV-to-UAV mesh with multi-hop routing to a gateway")

    # ------------------------------------------------- wireless building blocks
    for pname, pcls in PROPAGATION_MODELS.items():
        r.propagation.register(pname, pcls, doc=(pcls.__doc__ or "").strip().split("\n")[0])
    for aname, acls in ANTENNA_PATTERNS.items():
        r.antennas.register(aname, acls, doc=(acls.__doc__ or "").strip().split("\n")[0])
    for sname, scls in SCHEDULERS.items():
        r.schedulers.register(sname, scls, doc=(scls.__doc__ or "").strip().split("\n")[0])

    # --------------------------------------------------------------- formations
    for fname, ffn in FM.FORMATIONS.items():
        r.formations.register(fname, ffn, doc=(ffn.__doc__ or "").strip().split("\n")[0])

    # -------------------------------------------------------------------- tasks
    from ..envs.tasks import TASKS

    for name, cls in TASKS.items():
        r.tasks.register(name, cls)

    # ------------------------------------------------------------------- faults
    for cls in (
        F.RotorFailure,
        F.ThrustLoss,
        F.PayloadChange,
        F.ComShift,
        F.ActuatorStuck,
        F.BatteryDrop,
        F.SensorDropout,
        F.GpsOutage,
        F.SensorStuck,
        F.SensorBiasStep,
        F.SensorNoiseScale,
        F.SensorDelay,
        SF.SensorScaleError,
        SF.SensorSaturation,
        SF.SensorSpikes,
        SF.SensorDrift,
        SF.SensorNaN,
        SF.SensorDelaySpikes,
        F.WindDisturbance,
        F.WindGust,
        F.CommBlackout,
        GnssJamming,
        GnssSpoofing,
        NetworkJamming,
    ):
        r.faults.register(cls.name, cls, doc=(cls.__doc__ or "").strip().split("\n")[0])

    # ------------------------------------------------------------------- metrics
    from ..metrics import aggregate, flight_metrics

    r.metrics.register(
        "flight",
        lambda archive, skip=0.0: aggregate(flight_metrics(archive, skip=skip)),
        doc="mean over vehicles of the standard flight metrics (tracking error, energy, tilt, ...)",
    )

    # ------------------------------------------------------------------ swarm laws
    r.swarm_laws.register("formation_keeping", SL.FormationKeeping, doc="neighbours wait for a lagging vehicle: the shape holds under disturbances")
    r.swarm_laws.register("rendezvous", SL.Rendezvous, doc="average consensus on position over the lossy link")
    r.swarm_laws.register("coverage", SL.CoverageLaw, doc="Lloyd coverage of a rectangle on the believed positions", aliases=("lloyd",))

    # -------------------------------------------------------------------- hooks
    r.hooks.register("progress", H.ProgressHook, doc="print a progress bar")
    r.hooks.register("event_log", H.EventLogHook, doc="print events as they occur")
    r.hooks.register("stop_on_event", H.StopOnEventHook, doc="stop the run when an event matches")
    r.hooks.register("profiler", H.ProfilerHook, doc="per-task wall-time report")
    r.hooks.register("slung_load", EF.SlungLoadHook, doc="a payload hanging on an elastic rope (tension, swing, slack)")
    r.hooks.register("proximity_effects", EF.ProximityEffectsHook, doc="ceiling thrust gain and wall suction from nearby surfaces")
    r.hooks.register("battery_model", EF.BatteryModelHook, doc="detailed battery: Peukert capacity, polarisation, heating, ageing")

    # ------------------------------------------------------------ health checks
    for cls in HC.DEFAULT_CHECKS:
        r.health_checks.register(cls.name, cls, doc=(cls.__doc__ or "").strip().split("\n")[0])
    r.health_checks.register("geofence", GeofenceCheck, doc="geofences / no-fly zones: emits geofence.breach and geofence.clear events")

    for kind, scan_cls in SCAN_PATTERNS.items():
        r.scan_patterns.register(kind, scan_cls, doc=(scan_cls.__doc__ or "").strip().split("\n")[0])

    # ------------------------------------------------------------------- actors
    for kind, actor_cls in ACTORS.items():
        r.actors.register(kind, actor_cls, doc=(actor_cls.__doc__ or "").strip().split("\n")[0])


import aevion.registry as _pkg  # noqa: E402

_register_all(_pkg.registries)
