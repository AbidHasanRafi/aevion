# Aevion documentation

> One-file edition of the hand-written Aevion docs. The front page is [README.md](README.md); the generated API reference and figure gallery are built from the source with `python tools/gen_site.py`.

## Contents

- Getting started
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Architecture](#architecture)
- Simulation
  - [Mathematical models](#mathematical-models)
  - [Vehicle realism and aircraft](#vehicle-realism-and-aircraft)
  - [Environment](#environment)
  - [Sensors](#sensors)
  - [Advanced control](#advanced-control)
  - [State estimation](#state-estimation)
  - [Planning and avoidance](#planning-and-avoidance)
  - [Wireless](#wireless)
  - [Multi-agent coordination](#multi-agent-coordination)
  - [Bridges: MAVLink, PX4 logs, HIL and ROS 2](#bridges-mavlink-px4-logs-hil-and-ros-2)
  - [System identification](#system-identification-1)
- Machine learning, experiments and analysis
  - [Machine learning guide](#machine-learning-guide)
  - [Vision datasets](#vision-datasets-1)
  - [Experiments](#experiments)
  - [Analysis and research-grade figures](#analysis-and-research-grade-figures)
- Tooling and reference
  - [Viewer](#viewer-1)
  - [Run archives](#run-archives)
  - [Writing a plugin](#writing-a-plugin)
  - [Untrusted configurations](#untrusted-configurations)
  - [Configuration reference (schema version 2)](#configuration-reference-schema-version-2)
  - [Command line](#command-line)
- Validation and roadmap
  - [Validation and fidelity](#validation-and-fidelity)
  - [Roadmap to 1.0](#roadmap-to-10)
- Project
  - [Changelog](#changelog)
  - [Contributing to Aevion](#contributing-to-aevion)
  - [Code of conduct](#code-of-conduct)
  - [Security policy](#security-policy)
  - [Versioning and deprecation policy](#versioning-and-deprecation-policy)
  - [Releasing Aevion](#releasing-aevion)

---

<!-- page: installation -->

## Installation

Aevion needs Python 3.10 or newer. The core depends only on NumPy and SciPy.

```bash
pip install aevion                  # core
pip install "aevion[fast]"          # + Numba fused kernel (recommended on CPU)
pip install "aevion[viz]"           # + Matplotlib (figures, reports)
pip install "aevion[all]"           # everything optional
```

From a checkout:

```bash
git clone https://github.com/aevion/aevion && cd aevion
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Optional extras

| Extra | Adds | Needed for |
| --- | --- | --- |
| `fast` | Numba | 5-10x faster CPU physics for batches of vehicles |
| `torch` | PyTorch | GPU (CUDA) backend, differentiable dynamics, torch-native environments |
| `gym` | Gymnasium | `aevion.envs` environments, Stable-Baselines3 adapter |
| `viz` | Matplotlib | research-grade figures, HTML reports |
| `yaml` | PyYAML | YAML configuration files |
| `hdf5`, `parquet` | h5py, pyarrow | archive export formats |
| `docs` | MkDocs Material, mkdocstrings | building this documentation |

Nothing optional is imported until you use it, so a bare install starts in a fraction of a second.

### Check the installation

```bash
aevion info        # versions, backends, kernels, plugins
aevion demo        # fly a demo, write an archive + report + 3-D replay
```

`aevion info` reports which acceleration is active: the Numba kernel is verified against the reference dynamics on
first use and silently falls back to the reference path if it ever disagrees.

### GPU

Install a CUDA build of PyTorch following the instructions on pytorch.org, then pass `backend="cuda"`:

```python
from aevion import Multirotor
sim = Multirotor("generic_1kg", n=100_000, backend="cuda")
```

### Offline machines

Everything works offline. The only network access in Aevion is the optional `aevion replay --offline`, which downloads
two pinned three.js files over HTTPS once, so the replay file can embed them.

---

<!-- page: quickstart -->

## Quickstart

Code blocks preceded by an invisible marker are executed by the test-suite (`tests/test_docs.py`), so they stay
correct as the library evolves. Blocks on one page share state, top to bottom.

### Fly a configuration

A configuration is a plain dictionary (or a YAML/JSON file). Everything has a default; you write only what you change.

<!-- docs-test -->
```python
from aevion.engine import Simulation

config = {
    "schema_version": 2,
    "name": "quickstart",
    "seed": 1,
    "duration": 6.0,
    "trajectory": {"type": "figure8", "params": {"center": [0, 0, 2.0], "size": 2.0, "period": 6.0}},
    "wind": {"type": "constant", "params": {"velocity": [1.0, 0.0, 0.0]}},
    "sensors": ["imu"],
}
archive = Simulation(config).run()
print(archive)  # a one-line summary: steps, vehicles, time span, channels
```

The result is a run archive: named channels (`t`, `pos`, `vel`, `quat`, `rotor`, `ref_pos`, ...) with the full
configuration, provenance and integrity hashes.

<!-- docs-test -->
```python
import numpy as np

pos = archive["pos"]                 # (steps, vehicles, 3), world frame ENU, metres
print(pos.shape)
err = np.linalg.norm(archive["pos"][:, 0] - archive["ref_pos"][:, 0], axis=-1)
print(f"RMS tracking error: {np.sqrt((err ** 2).mean()) * 100:.1f} cm")
assert err.max() < 0.5
```

Conventions: world frame ENU (z up), body frame FLU, quaternions `[w, x, y, z]` body to world, SI units.

### Validate before you run

A typo is reported with its path and a suggestion, all problems at once:

<!-- docs-test -->
```python
from aevion.config import load_config
from aevion.errors import ConfigError

try:
    load_config({"duration": 5, "sensors": [{"type": "gps", "params": {"rat": 5}}], "wnid": {"type": "dryden"}})
except ConfigError as e:
    print(e)
    assert len(e.issues) == 2
```

From the shell: `aevion validate sim.yaml`. Run `aevion schema` for a JSON Schema that gives you completion and
checking in editors.

### Inject a fault

Faults are part of the configuration. This one cuts the GPS for four seconds and fails a rotor at t = 4 s:

<!-- docs-test -->
```python
faulty = {
    **config,
    "duration": 8.0,
    "sensors": ["imu", "gps"],
    "estimator": {"type": "mahony_kf", "handover_s": 1.0},
    "faults": [
        {"type": "gps_outage", "t_start": 1.5, "t_end": 5.5},
        {"type": "rotor_failure", "t_start": 4.0, "rotor": 1, "severity": 0.3},
    ],
    "health": {"on_diverge": "stop"},
}
result = Simulation(faulty).run()
print([e["type"] for e in result.meta["events"]][:6])
```

Health monitors watch for divergence, NaNs and saturation; `on_diverge` chooses between raising, stopping cleanly with
the partial archive, or warning.

### Save, reload, replay, compare

<!-- docs-test -->
```python
import tempfile
from pathlib import Path

from aevion.datasets import RunArchive

with tempfile.TemporaryDirectory() as d:
    path = archive.save(Path(d) / "run.aevd")             # chunked, hash-verified directory store
    again = RunArchive.open_store(path)                    # lazy: channels are read from disk on demand
    assert again.verify_integrity() == []   # an empty list means every chunk hash matches
    print(sorted(again.channels)[:5])
    again.release()
```

```bash
aevion run sim.yaml --stream --open      # stream to disk while running, then open the 3-D replay
aevion inspect run.aevd                  # summary and metrics
aevion verify run.aevd                   # re-simulate and compare bit for bit
aevion diff run_a.aevd run_b.aevd        # what changed, channel by channel
```

### Make a figure

<!-- docs-test -->
```python
import matplotlib

matplotlib.use("Agg")
from aevion.analysis import figures as F

fig = F.fig_tracking(archive)            # research-grade figure, single column
print(fig.get_size_inches())
```

`fig.savefig("tracking.pdf")` gives a vector figure that can go into a LaTeX paper; see the gallery.

### Scale up

The same configuration with a thousand vehicles:

<!-- docs-test -->
```python
swarm = {"duration": 1.0, "vehicle": {"count": 200, "formation": {"type": "grid", "params": {"spacing": 2.0}}}}
big = Simulation(swarm).run()
print(big.n_vehicles, "vehicles,", big.n_steps, "steps")
```

On a GPU set `engine: {backend: cuda}`; on CPU install the `fast` extra for the fused Numba kernel.

### Next

- [Configuration reference](#configuration-reference-schema-version-2): every field, its type, unit and default.
- [Machine learning](#machine-learning-guide): Gymnasium and vectorised environments.
- [Plugins](#writing-a-plugin): add your own sensors, controllers and faults.

---

<!-- page: architecture -->

## Architecture

Aevion is organised in layers. Each layer depends only on the ones above it in this list.

```text
 config      typed, validated, versioned configuration (schema v2) ─ migrates legacy scenarios
 registry    name → factory for every component kind, plus plugin discovery
 engine      Simulation: scheduler, event bus, hooks, fault timeline, RNG streams, health, recording
 components  dynamics (kernels) · controllers · sensors · estimators · networks · swarm · world/wind
 datasets    chunked run store (v2), v1 files, provenance, run registry, diff, export
 analysis    metrics, figures, reports, statistics          viewer   3-D replay
```

### The tick

One *tick* is one physics step (`engine.dt`, default 5 ms). Periodic tasks run in priority order inside the
tick; the physics step then advances the vehicles.

| priority | task | what it does | rate |
|---:|---|---|---|
| 5 | `faults` | start/stop scheduled faults (`FaultTimeline.update`) | every tick |
| 10 | `sense` | `SensorSuite.update`: each sensor applies its own rate, latency and dropout | every tick |
| 20 | `estimate` | sensor fusion; after `handover_s` the controller flies on the estimate | every tick |
| 30 | `mission` | reference from the trajectory (+ swarm collision-avoidance offsets) | every tick |
| 40 | `control` | controller → rotor command, held (zero-order) between control ticks | `controller_rate_hz` |
| 80 | `network` | radio-link evaluation | `network.rate_hz` or the record rate |
| 90 | `record` | one archive row | `record_every` / `logging.rate_hz` |
| 91 | `record_sensors` | store each fresh sample of recorded sensors | every tick |

Then: `pre_step` hooks (which may replace `sim.ctx.cmd`) → fault command modifiers → physics (`dyn.step`) → event
detection (`ground.impact`, `crash`, `collision`, `battery.low`) → health checks → `post_step` hooks.

Scheduling is by integer ticks (`rate_hz` is converted to the nearest whole number of ticks, with a warning if it
is more than 1 % off), so runs are exactly reproducible and free of floating-point drift. `aevion run --profile`
prints a per-task wall-time table.

### Determinism

* Every stochastic component owns its generator. Sensors, wind, the network and the swarm keep a fixed seeding scheme
  (so legacy results are reproduced bit for bit; see `tests/test_engine.py`);
  new components should draw from `sim.rng.stream("name")` (`RngStreams`), whose streams are independent and do not
  change when other components are added or removed.
* `aevion verify run.aevd` rebuilds the simulation from the embedded configuration and compares every channel.

### Components and the registry

Everything a configuration can name is looked up in a registry (`aevion.registry.registries`):
`vehicles, controllers, trajectories, sensors, estimators, winds, worlds, channels, networks, tasks, planners,
formations, faults, hooks, health_checks, metrics`. `aevion registry <kind>` lists them. Unknown names produce
"did you mean …" errors; unknown *parameters* are caught at validation time by introspecting the factory signature.
Third-party packages register components through the `aevion.plugins` entry point; see [Writing a plugin](#writing-a-plugin).

### Dynamics kernels

`dynamics/multirotor.py` contains the model *definition* (backend-generic: NumPy, torch CPU/CUDA, differentiable).
`dynamics/fast.py` is a fused, multi-threaded Numba re-implementation for the NumPy backend. They are interchangeable
kernels (`engine.kernel: auto | reference | numba`). A parity check runs both on randomised states the first time a
configuration uses the fast kernel (≈30 ms, cached); if they disagree, `auto` falls back to the reference with a
warning and an explicit `numba` raises. `tests/test_fast_parity.py` also pins them to 1e-8 across every
integrator, vehicle, gravity/atmosphere model, terrain, collision, failure and battery mode.

### Faults

A `Fault` has `on_start` / `on_end` (revert) and may modify the rotor command while active. Built-in: rotor failure,
thrust loss, payload change, centre-of-mass shift, stuck actuator, battery drop, sensor dropout / stuck / bias step /
noise scale / delay, GPS outage, wind disturbance and gust, communication blackout. Every fault is reversible, so
transient faults (`t_start`…`t_end`) behave correctly, and each emits `fault.start` / `fault.end` events that appear
in the archive and the viewer timeline.

### Hooks and events

`Hook` subclasses (or `CallbackHook(pre_step=…)`) receive `on_start, on_reset, pre_step, post_step, on_event, on_end`.
Errors inside a hook name the hook and the step; `HookManager(isolate=True)` disables a failing hook instead. The
event bus (`sim.bus`, `sim.emit`) supports glob subscriptions (`"fault.*"`).

### Health

`HealthMonitor` runs `HealthCheck` objects (finiteness every tick; quaternion norm, speed, body rate, altitude and rotor
speed every 5; battery and saturation every 10). A hard violation follows the policy: `raise` →
`SimulationDiverged(step, t, vehicle, quantity, last_good)`, `stop` → the engine ends the run cleanly, `warn`. Soft
violations warn once. With streaming archives a failed run leaves a readable archive marked `diverged`.

### Frames and time

`aevion.frames` provides `Transform` (quaternion algebra, Shepperd conversions, slerp), `TransformTree` (static and
time-varying frames, cycle detection, order-independent serialisation) and `ClockModel` (offset, drift, jitter). Sensors
accept `extrinsics` (pose in the body frame; rotates/offsets LiDAR rays, cameras and rangefinders) and `clock`.

### Archives

See [Run archives](#run-archives): chunked streaming store with hashes, provenance, run registry, diff, Parquet export.

---

<!-- page: models -->

## Mathematical models

All quantities SI. World frame ENU (`x` east, `y` north, `z` up). Body frame FLU (`x` forward, `y` left,
`z` up). `R = R(q)` rotates body→world; quaternions are `[w, x, y, z]` (Hamilton).

### 1. Rigid-body dynamics (`dynamics/multirotor.py`)

For each vehicle (batched over `N`):

```
p'   = v
m v' = m g(z) ẑ·(−1) + R ( F_thrust + F_drag + F_rotor-drag )
q'   = ½ q ⊗ [0, ω]
J ω' = τ_thrust + τ_reaction + τ_gyro + τ_damp − ω × (J ω + h_r)
w_i' = (w_cmd,i − w_i) / τ_m                 (τ_up / τ_down selectable)
```

#### Rotors

With rotor speed `w_i` (rad/s), spin `s_i = ±1` (+1 counter-clockwise from above), health `η_i ∈ [0,1]` and
air-density ratio `σ = ρ(z)/ρ₀`:

```
T_i = kT σ η_i w_i² · f_inflow · f_ground          thrust along +z_b
Q_i = kQ σ η_i w_i²                                aerodynamic reaction torque
τ_thrust   = Σ (r_i − r_com) × (T_i ẑ)
τ_reaction = −Σ s_i (Q_i + J_r w_i') ẑ             (drag torque + rotor spin-up)
h_r        = Σ s_i J_r w_i ẑ ,   τ_gyro = −ω × h_r
```

* Ground effect (Cheeseman & Bennett): `f_ground = 1 / (1 − (R_p / 4z_agl)²)`, clipped to `[1, 1.5]`.
* Inflow loss (optional): `f_inflow = clip(1 − c · V_c / (w_i R_p), 0, 1.5)`, `V_c` = climb rate relative to air.
* Allocation matrix `A` (4 × M) maps `w²` to `[T, τx, τy, τz]`:
  `A = [kT; y_i kT; −x_i kT; −s_i kQ]`. Hover speed `w_h = √(m g / (M kT))`.

#### Aerodynamic drag

Relative air velocity in body axes `v_r = Rᵀ(v − w_wind)`:

```
F_drag       = −(c_lin + c_quad |v_r|) ⊙ v_r · σ            per body axis
F_rotor-drag = −k_rd (Σ w_i) [v_r,x, v_r,y, 0]               induced/blade-flapping-like drag
τ_damp       = −C_ω ω
```

#### Battery and power

Mechanical power `P_m = Σ Q_i w_i`; electrical `P = P_m/η_ESC + P_avionics`. State of charge integrates `P`
over the pack capacity. With `voltage_effects` the pack follows `V_oc(SoC) = n(3.3 + 0.9 SoC^0.8)` with series
resistance `R`: terminal voltage `V = ½(V_oc + √(V_oc² − 4RP))`, and the available maximum rotor speed scales
with `V / V_full`.

#### Contact and collisions

* Ground (flat or bilinear heightmap): inelastic with restitution `e`, Coulomb friction `μ`
  (`v_xy ← max(0, 1 − μ g Δt/|v_xy|) v_xy`), angular damping while in contact.
* Obstacles (boxes, spheres, vertical cylinders): the vehicle is a sphere of radius `r_c`; penetration depth and
  normal are summed over obstacles, position is projected out and the normal velocity reflected with `e`.

#### Integration

`rk4` (default), `midpoint`, `semi_implicit`, `euler`; quaternions are renormalised after each stage. The motor
lag is stiff (`τ_m` ≪ `Δt` makes explicit RK4 unstable), so `Multirotor` automatically splits a step into
`n_sub` sub-steps such that `Δt_sub ≤ 1.5 · 0.5 τ_m` (RK4/midpoint) or `0.5 · 0.5 τ_m` (Euler).
The command is held constant over the step (zero-order hold).

### 2. Environment (`core/constants.py`, `environments/`)

* Gravity: constant `g₀ = 9.80665`; inverse-square `g₀ (Rₑ/(Rₑ+h))²`; WGS-84 Somigliana latitude model with
  free-air correction `g_φ − 3.086·10⁻⁶ h`; custom (Moon 1.62, Mars 3.71).
* Atmosphere: ISA troposphere, `T = T₀ − Lh + ΔT`, `p = p₀(T_std/T₀)^{g/(RL)}`, `ρ = p/(R T)`.
* Wind: constant, boundary-layer shear (`u_ref (z/z_ref)^α` or log law), discrete 1−cos gust, and
  Dryden turbulence with the MIL-F-8785C low-altitude scale lengths and intensities
  (`σ_w = 0.1 W₂₀`, `L_w = h`, `L_u = L_v = h/(0.177+0.000823h)^1.2` in feet). Shaping filters are driven by
  white noise normalised so the output variance equals `σ²`.

### 3. Control (`control/`)

* Geometric SE(3): `F = m(−ω_p² e_p − 2ζω_p e_v − k_i ∫e_p + a_ref + g ẑ)`; thrust `T = F·b₃`; desired attitude
  `R_d = [b₂×b₃, b₂, b₃]` with `b₃ = F/|F|`, `b₂ = b₃×b₁c / |·|`; `e_R = ½(R_dᵀR − RᵀR_d)^∨`;
  `τ = J(−ω_R² e_R − 2ζω_R e_Ω) + ω × Jω`. Gains are natural frequencies, derived from the motor lag.
* Cascaded PID: position P → velocity PI → attitude P (on `e_R`) → rate P/PI, with tilt and speed limits.
* LQR: 12-state small-angle hover model with mass/inertia-normalised inputs; Riccati via SciPy; acceleration
  feedforward through the equilibrium tilt.
* Allocation: pseudo-inverse with collective-shift desaturation (attitude over altitude), or the
  fault-tolerant bounded least-squares allocator (healthy rotors only, yaw de-prioritised).

### 4. Sensors (`sensors/`)

* IMU: `a = s ⊙ f_b + b_a + n_a/√Δt`, `ω = s ⊙ ω_b + b_g + n_g/√Δt`, biases random-walk with `σ_bw √Δt`,
  optional low-pass and saturation. `f_b = Rᵀ(a − g)` is the specific force.
* GPS: position error = first-order Gauss–Markov (`τ ≈ 60 s`) + white noise, velocity white noise, latency.
* Barometer: ISA pressure + noise + drift → altitude by the inverse ISA formula.
* Magnetometer: 50 µT Earth field with inclination/declination, hard-iron bias, noise.
* Ray casting: slab method for boxes, quadratic for spheres, side + caps for cylinders, ray-marched
  bisection for heightmaps. Numba (multi-core) for NumPy inputs; vectorised implementation for torch tensors.
* Depth is z-depth (range · cos of off-axis angle); depth noise `σ = a + b z²`.

### 5. Wireless (`wireless/`)

Path loss `PL = FSPL(d,f) + η_{LoS|NLoS}` (+ log-normal shadowing); LoS from the Al-Hourani probability
`P_LoS(θ) = 1/(1 + a e^{−b(θ−a)})` or from ray casting against the world geometry; elevation-dependent
Rician `K`; SINR with all other cells as interference; `R = B log₂(1 + SINR/Γ)`; down-tilted 3GPP sector
pattern `A(θ) = −min(12((θ−θ_tilt)/θ_3dB)², SLA)`.

### References

* T. Lee, M. Leok, N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on SE(3)", CDC 2010.
* R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear complementary filters on the special orthogonal group", TAC 2008.
* A. Al-Hourani, S. Kandeepan, S. Lardner, "Optimal LAP altitude for maximum coverage", WCL 2014.
* MIL-F-8785C, "Flying qualities of piloted airplanes", 1980 (Dryden turbulence).
* I. C. Cheeseman, W. E. Bennett, "The effect of the ground on a helicopter rotor in forward flight", 1955.
* 3GPP TR 36.777, TR 36.814.

---

<!-- page: vehicles -->

## Vehicle realism and aircraft

This section covers physical rotor components (propeller, motor, ESC, battery) that a vehicle can be built from, loads and proximity
effects that act on the airframe, and fixed-wing and VTOL aircraft.

### Propeller, motor, ESC and battery

Propeller (`aevion.vehicles.propeller`). A blade-element momentum model (BEMT): the blade is divided into stations, each
solved for the induced inflow with Prandtl tip-loss and hub-loss factors, using airfoil lift and drag that stay defined at every
angle of attack (attached, stalled, and flat-plate beyond). `get_propeller("10x4.5")` returns one of the built-in designations
(diameter × pitch in inches, like `5x4.3x3` for three blades); `static(rpm)` and `operating_point(rpm, airspeed)` return thrust,
torque, power, `CT`, `CP`, advance ratio, efficiency and figure of merit. A static 10-inch propeller gives a figure of merit
near 0.58, in the range reported for small fixed-pitch propellers; thrust scales with rpm², power with rpm³, more blades and
more pitch give more thrust and power, and the tests check those laws and the shape of the advance-ratio curves.

> Warning: The propeller data is generic
>
> The blade geometry (chord and pitch distribution) and the airfoil polars are synthetic, chosen to reproduce typical
> small-propeller coefficients; they are *not* measurements of any commercial propeller. For a specific one, measure
> `CT(J)` and `CP(J)` (or use manufacturer or UIUC data) and construct a `Propeller` with your own figures.

Motor and ESC (`aevion.vehicles.motor`). `BLDCMotor(kv, resistance, i0, rotor_inertia)` solves the motor equations
`V = IR + ω/Kv` and `τ = (I − I₀)/Kv` together with the propeller's torque curve for the operating point (a closed form for a
quadratic load, a root solve otherwise). `ESC` scales the voltage by a throttle curve (`min_duty`, `expo`) with a conversion efficiency; the motor model's current limit
caps both the current and the torque. `MotorPropSystem` joins them: `hover(mass, n_rotors, voltage)` gives the electrical power,
efficiency and throttle needed to hold a weight, `thrust` and `max_speed` give the envelope. The spin-up time constant matches the
linearised ODE in the tests, and the motor identities (power in = mechanical + copper + iron/no-load losses) balance.

Vehicles from components. `VehicleSpec.from_components("10x4.5", mass=1.2, arm=0.22)` derives `kT`, `kQ`, the rotor inertia,
maximum rotor speed and the battery parameters from the propeller and motor, then validates the result
(`spec.validate()` returns problems). The vehicle hovers at the throttle the model predicts, which is a good consistency check
of the pieces.

<!-- docs-test -->
```python
import math
from aevion.dynamics import Multirotor
from aevion.vehicles.motor import BLDCMotor, MotorPropSystem
from aevion.vehicles.propeller import get_propeller
from aevion.vehicles.spec import VehicleSpec

prop = get_propeller("10x4.5")
lo, hi = prop.static(3000.0), prop.static(6000.0)
assert abs(hi.T / lo.T - 4.0) < 0.2                                                   # thrust ~ rpm^2 (doubling rpm quadruples thrust)
assert 0.5 < lo.figure_of_merit < 0.7                                                 # typical of small propellers

spec = VehicleSpec.from_components("10x4.5", mass=1.2, arm=0.22)
assert spec.validate() == [] and 1.5 < spec.max_thrust() / (spec.mass * 9.80665) < 4.5
motor = MotorPropSystem(BLDCMotor(kv=920.0, resistance=0.09, i0=0.6, rotor_inertia=1.5e-5), prop)
hover = motor.hover(1.2, 4, 11.1)
assert 60.0 < hover["total_power"] / 1.2 < 220.0                                      # W/kg, the range of real multirotors

sim = Multirotor(spec, n=1)
st = sim.reset(pos=(0.0, 0.0, 2.0))
for _ in range(300):
    st = sim.step(st, sim.hover_command())
assert abs(st.pos[0, 2] - 2.0) < 0.02                                                 # the derived kT and mass are consistent
```

Battery pack (`aevion.vehicles.battery_pack`). `BatteryPack(PackSpec(...))` extends the first-order Thevenin pack with a
Peukert capacity correction for high discharge, a temperature-dependent internal resistance and capacity (cold packs deliver
less), a lumped thermal model (self-heating by `I²R`, cooling to ambient), and ageing (capacity fade and resistance growth with
cycles). It converts a *power* demand to a current by solving the quadratic of the terminal voltage, and flags a brownout
when the demand cannot be met. The energy balance (chemical energy = terminal energy + heat + stored change) is a test, not
an assumption. Enable it in a run with the `battery_model` hook:

```yaml
hooks: [{type: battery_model, params: {capacity_ah: 5.0, ambient_c: 20.0}}]
```

### Loads and proximity effects

Both are engine hooks that act through the external-wrench channel, a force and torque in the world frame that the dynamics
kernels (the fused numba kernel and the reference kernel, which are parity-tested) add to the airframe.

* `slung_load`: a point-mass payload on an elastic rope (spring-damper, tension only, with quadratic drag on the load); it
  behaves as a spherical pendulum while the rope is taut. The load swings as a pendulum of the right period, the static tension equals the weight, and energy is conserved
  in the undamped case. The vehicle feels the tension as a force *and* the torque of an off-centre attachment point.
* `proximity_effects`: a ceiling effect (thrust rises under a ceiling, in the form of the Cheeseman–Bennett ground-effect law
  with a ceiling constant) and a wall effect (suction towards a nearby wall, sensed with rays against the world). Ground effect
  is already part of the dynamics. They are first-order corrections of the right size from the literature, not a CFD result.

```yaml
hooks:
  - {type: slung_load, params: {mass: 0.3, length: 1.5}}
  - {type: proximity_effects}
```

### Fixed-wing aircraft

`aevion.aircraft.FixedWing` is the six-degree-of-freedom fixed-wing model of Beard and McLain, *Small Unmanned Aircraft: Theory
and Practice* (2012): lift and drag with a post-stall blend (linear lift curve, then a flat-plate model), side force, the three
moments with the rate derivatives, propeller thrust, wind and gust input, and the NED/FRD formulas converted at the boundary so the
state follows Aevion's ENU/FLU conventions. `AEROSONDE` is the book's parameter set (an 11 kg aircraft).

`trim(model, airspeed, gamma=..., radius=...)` finds the controls for steady straight, climbing or turning flight; `linearize`
returns the state-space matrices at the trim point; `modes` extracts the classical modes (short period, phugoid, Dutch roll,
roll subsidence, spiral) with natural frequency, damping and time constants. `Autopilot` is the successive loop-closure design of
the book (roll and pitch attitude, course, altitude, airspeed), and `PathFollower` follows straight lines and orbits.

<!-- docs-test -->
```python
from aevion.aircraft import FixedWing
from aevion.aircraft.trim import linearize, modes, trim

fw = FixedWing(None, 1, 0.01)                       # the Aerosonde of the book
t = trim(fw, 35.0)
assert t.converged and abs(t.alpha) < 0.05 and abs(t.gamma) < 1e-12                  # level flight at a small angle of attack
m = modes(linearize(fw, t))
assert m["short_period"]["wn"] > 5 * m["phugoid"]["wn"]                              # the two longitudinal modes are well separated
assert 0.2 < m["dutch_roll"]["zeta"] < 0.8 and m["roll"]["eigenvalue"] < -10.0      # lightly damped Dutch roll, fast roll subsidence
```

### VTOL aircraft

`QuadPlane` (lift-plus-cruise: four lift rotors and a wing with a pusher) and `TiltRotorPlane` (the same four rotors tilt from
vertical to horizontal, with no pusher) add rotor mechanics to the fixed-wing model. `QuadPlaneController` and `TiltRotorController` blend multirotor and fixed-wing control
authority with a factor λ that depends on airspeed, and use a control allocation with priority desaturation (roll and pitch are
protected over yaw and collective when a motor saturates). They fly a hover → forward flight → hover transition in the tests.

> Warning: What these aircraft models are and are not
>
> They are reduced-order models from the textbook literature, not validated against any real airframe. Aerodynamics
> are linear-in-coefficients with a simple post-stall blend; there is no propeller slipstream over the wing, no rotor–wing
> interference, the rotors are point thrusts with first-order lags (no gyroscopic rotor torque). Use them for
> controller and estimator development, transition logic, and teaching, not for predicting the performance of a
> particular aircraft.

---

<!-- page: environment -->

## Environment

The environment adds geographic coordinates, sun position, weather that changes air density and visibility, imported buildings and
terrain, wind around obstacles, moving obstacles and geofences. All of it is optional and off by default, so existing configurations
and archives behave as before (the configuration hash of a configuration without an `environment:` block is unchanged).

Conventions: the world frame is ENU (x east, y north, z up). A geographic origin ties that frame to latitude / longitude /
altitude; without one the world is a flat local frame.

| You want | Use |
| --- | --- |
| Real coordinates, sun, magnetic field | `environment.origin`, `environment.utc` |
| Rain, fog, heat, pressure | `environment.weather` (preset and/or fields) |
| Buildings from a map or a CAD model | `world: {type: scene, params: {buildings: …, meshes: […]}}` |
| Real terrain | `world.params.terrain` (GeoTIFF, ASCII grid, `.npz`, or a heightmap) |
| Wind that respects buildings and height | `wind: urban`, `log_law` or `gridded` |
| Birds, aircraft, vehicles | `environment.actors` |
| No-fly zones and keep-in volumes | `environment.geofences` |
| Distance fields for planners | `SignedDistanceField.from_world(world)` |

### Coordinates, time and the sun

<!-- docs-test -->
```python
import numpy as np
from aevion.environments.geodesy import LocalFrame, vincenty_inverse
from aevion.environments.solar import sun_position, sunrise_sunset
from aevion.environments.geomagnetic import TiltedDipole

zurich = LocalFrame(47.3769, 8.5417, 408.0)
east_100m = zurich.to_lla([100.0, 0.0, 0.0])                    # (lat, lon, alt)
back = zurich.to_enu(east_100m)
assert np.allclose(back, [100.0, 0.0, 0.0], atol=1e-6)

# geodesic distance on the WGS84 ellipsoid (Vincenty); returns metres and both azimuths
dist_m, az_fwd, az_back = vincenty_inverse(47.3769, 8.5417, 48.8566, 2.3522)
assert 485e3 < dist_m < 495e3                                   # Zurich -> Paris

sun = sun_position(47.3769, 8.5417, "2026-06-21T10:00:00Z")      # NOAA solar ephemeris
assert 55.0 < float(sun["elevation_deg"]) < 70.0
rise, sets = sunrise_sunset(47.3769, 8.5417, "2026-06-21")
assert rise is not None and sets is not None and sets > rise

b = TiltedDipole().field_enu(47.3769, 8.5417, 408.0, year=2026.0)   # microtesla, ENU
assert 35.0 < float(np.linalg.norm(b)) < 60.0 and b[2] < 0           # points down in the northern hemisphere
```

The geodesy module implements WGS84 (ECEF, geodetic, local ENU), haversine and Vincenty distances, bearings and the principal radii
of curvature. `sun_position` follows the NOAA ephemeris (about 0.01° in the modern era); `sunrise_sunset` includes refraction.

> Note: Magnetic field
>
> The default model is a tilted dipole built from the degree-1 IGRF coefficients (with secular variation). It captures the
> global pattern (inclination, intensity from about 25 to 65 µT) but not the regional anomalies of the full IGRF / WMM, which
> typically differ by several degrees of declination. Use `FixedField` for a fixed, known field, or write a `MagneticModel`
> that wraps a spherical-harmonic table; the magnetometer accepts any of them.

### Weather

<!-- docs-test -->
```python
from aevion.environments.weather import Weather, moist_air_density, fog_extinction

assert abs(moist_air_density(101325.0, 15.0, 0.0) - 1.225) < 0.002          # dry ISA sea level
assert moist_air_density(101325.0, 35.0, 0.9) < moist_air_density(101325.0, 35.0, 0.0)   # humid air is lighter
fog = Weather.preset("fog")
assert fog.visibility_m == 200.0 and fog_extinction(fog.visibility_m) > 0.01
```

Presets: `clear`, `overcast`, `rain`, `heavy_rain`, `fog`, `hot`, `cold`, `storm`; any field can be overridden
(`weather: {preset: rain, rain_mm_h: 6}`). Weather acts on the vehicle through the air density (temperature, sea-level pressure
and humidity enter the moist-air gas law; the fused kernel reproduces the reference model exactly) and on the sensors through
extinction: LiDAR and camera see rain and fog, GNSS and radar are only mildly affected. Rain also scales the viewer's fog and haze.

### Buildings, models and terrain

<!-- docs-test -->
```python
import numpy as np
from aevion.environments.mesh import TriangleMesh
from aevion.environments.world import World

tower = TriangleMesh.box(center=(20, 0, 15), size=(8, 8, 30))
assert tower.is_watertight() and abs(tower.volume - 8 * 8 * 30) < 1e-6

world = World()
world.add_mesh(tower)
dist = world.distance_to_nearest(np.array([[10.0, 0.0, 15.0]]))    # unsigned distance to the nearest surface
assert abs(float(dist[0]) - 6.0) < 1e-6                              # 6 m from the tower wall
inside = world.mesh_index.signed_distance(np.array([[20.0, 0.0, 15.0]]))
assert float(inside[0]) < 0                                          # negative inside a closed mesh
```

A `TriangleMesh` is an indexed triangle soup with a BVH (numba-compiled traversal, no allocation in the parallel loops). The
vehicle collides with meshes through the fused kernel, and every ray-casting sensor sees them. Loaders, all hardened for untrusted
files (size and triangle caps, bounds-checked indices, no pickles, no external references outside the file's directory):

| Format | Call |
| --- | --- |
| Wavefront OBJ | `TriangleMesh.from_obj(path, up="y")` |
| STL (ASCII and binary) | `TriangleMesh.from_stl(path)` |
| glTF 2.0 / GLB | `TriangleMesh.from_gltf(path, up="y")` |
| any of the above | `TriangleMesh.load(path)` (by extension) |
| OSM / GeoJSON footprints | `load_footprints(source, frame)` then `buildings_to_meshes` or `buildings_to_boxes` |
| GeoTIFF, ESRI ASCII grid, `.npz` | `load_terrain(path, frame, resolution, extent_m)` |

Footprint heights come from `height`, `building:levels` (times `level_height`) or `default_height`. Terrain voids are filled by
diffusion and the grid is resampled to a metric heightmap about the origin.

In a configuration, the `scene` world assembles all of this. File names (`buildings.file`, `terrain.file`, `meshes[].file`) resolve next to
the configuration file when it is loaded from disk (`load_config(path)`, `aevion run`), otherwise against the working directory; see
`examples/scenarios/urban_environment.yaml` and `examples/13_environment_tour.py`. An archive stores the configuration, not the files, so
`aevion verify` re-simulates a run only where those files can still be found; keep them beside the archive or use absolute paths.
`environment.origin` is also the anchor for `buildings` and `terrain`.


<!-- docs-test -->
```python
from aevion.engine import Simulation

cfg = {
    "duration": 3.0,
    "trajectory": {"type": "hover", "params": {"pos": [0, 0, 6.0]}},
    "world": {
        "type": "scene",
        "params": {
            "meshes": [{"type": "box", "center": [14, 0, 5], "size": [5, 5, 10], "color": [0.9, 0.3, 0.3]}],
            "boxes": [{"center": [-10, 5, 3], "size": [4, 4, 6]}],
        },
    },
    "environment": {
        "origin": {"lat": 47.3769, "lon": 8.5417, "alt": 408.0},
        "utc": "2026-06-21T10:00:00Z",
        "weather": {"preset": "overcast"},
    },
}
archive = Simulation(cfg).run()
assert archive.world.n_mesh_triangles == 12 and archive.world.origin.lat == 47.3769
```

### Wind that sees the world

| Type | Model | Parameters |
| --- | --- | --- |
| `log_law` | neutral atmospheric boundary layer, `u(z) = u*/κ · ln((z-d)/z0)` | `speed_ref`, `height_ref`, `z0`, `displacement`, `direction_deg` |
| `urban` | log law plus a Fackrell cavity wake behind each building (reduced speed, recirculation, turbulence) | as above, plus `buildings` (taken from the world when omitted) |
| `gridded` | trilinear (and temporal) interpolation of a precomputed field, e.g. exported from a CFD run | `data`, `origin`, `spacing`, `times` |

They compose with `dryden`, `gust` and `shear` through `composite`. `fackrell_reattachment_length(height, width, length)` exposes the
cavity length used by `urban`.

> Warning: Not CFD
>
> `urban` is an engineering wake model: it reproduces the first-order effects (shelter behind buildings, speed-up with height,
> a recirculation zone) and not separation bubbles, vortex shedding or street-canyon channelling. For those, run your own CFD and
> feed the result through `gridded`.

### Moving obstacles and geofences

<!-- docs-test -->
```python
from aevion.engine import Simulation

cfg = {
    "duration": 6.0,
    "trajectory": {"type": "hover", "params": {"pos": [0, 0, 6.0]}},
    "environment": {
        "actors": [{"type": "circular", "params": {"center": [0, 0, 6.0], "path_radius": 8.0, "omega": 0.5, "radius": 0.4}}],
        "geofences": [
            {"name": "yard", "kind": "inclusion", "zone": {"type": "cylinder", "center": [0, 0], "radius": 40.0, "z_max": 60.0}},
            {"name": "mast", "kind": "exclusion", "margin": 1.0, "zone": {"type": "cylinder", "center": [2, 0], "radius": 3.0}},
        ],
    },
    "health": {"checks": ["finite"]},
}
archive = Simulation(cfg).run()
assert archive["actor_pos"].shape[1:] == (1, 3)                       # the actor's recorded position
assert any(e["type"].startswith("geofence.") for e in archive.meta["events"])   # the hover point is inside the mast zone
```

Actors (`linear`, `circular`, `waypoints`, `bouncing`, `random_walk`) move kinematically and the vehicle collides with them. Zones are
`sphere`, `cylinder`, `box`, `polygon` and `altitude`; a fence is an `exclusion` (keep out) or an `inclusion` (stay inside), with a
`margin`. Breaches and clearances are events (`geofence.breach`, `geofence.clear`) with the fence name and signed distance, so the
viewer, the health monitor and your analysis all see them; set `geofence_stop` to end the run at the first breach.

### Distance fields for planners

<!-- docs-test -->
```python
import numpy as np
from aevion.environments.sdf import SignedDistanceField
from aevion.environments.world import World

w = World()
w.add_sphere((0, 0, 5), 2.0)
field = SignedDistanceField.from_world(w, bounds=(-6, -6, 0, 6, 6, 10), spacing=0.5, ground=False)
d = field.distance(np.array([[5.0, 0.0, 5.0]]))
assert abs(float(d[0]) - 3.0) < 0.5                                    # 3 m from the surface, within a voxel
assert float(field.distance(np.array([[0.0, 0.0, 5.0]]))[0]) < 0       # inside the sphere
```

`SignedDistanceField` stores a signed distance (negative inside obstacles) on a voxel grid with trilinear interpolation of the value
and its gradient, which is what MPC, CHOMP-style planners and control-barrier functions need.

### Viewer

The viewer draws meshes, geofence volumes (red keep-out, green keep-in, with labels) and moving obstacles, sets the sky to the local
solar time of the run's UTC stamp and origin, and closes the fog in with the weather's visibility. Live runs stream the same
information. Very large meshes (over 250 000 triangles in total) are left out of the replay and counted in
`payload["world"]["meshes_skipped"]`; they stay in the simulation.

### Limits

* Meshes are static (actors are primitives); there are no deformable or articulated obstacles.
* Imported models are used as given: self-intersections and inverted normals are *not* repaired (use `is_watertight()` and
  `degenerate_faces()` to check them).
* The magnetic model is a tilted dipole, the satellite almanacs are synthetic (see [Sensors](#sensors)), and `urban` wind is an
  engineering wake model.

---

<!-- page: sensors -->

## Sensors

The sensor models reproduce the error structure of real sensors and cover those a research
vehicle typically carries. Every sensor is a registered component: name it in `sensors:` (with `params:` and `record: true`), or
instantiate the class directly. They all see the same world, so a building that blocks the radar also shadows GNSS satellites and
appears in the LiDAR, and the weather that thins the air also attenuates the camera.

| Sensor | Registered as | What it models |
| --- | --- | --- |
| IMU | `imu` | Allan-variance noise (white, 1/f bias instability, random walk), vibration aliasing, temperature drift, g-sensitivity, misalignment |
| Magnetometer | `mag` | hard/soft iron, noise, and a position-dependent field from `TiltedDipole` |
| GNSS receiver | `gnss` | pseudoranges to a satellite constellation, weighted least squares, DOP, NLoS bias, jamming; `gps` keeps the simple model |
| Barometer, pitot, optical flow, ultrasonic, UWB | `baro`, `pitot`, `optical_flow`, `ultrasonic`, `uwb` | air data with lag and dynamic-pressure effects; flow with texture/illumination limits; cone-beam sonar with speed-of-sound error; two-way ranging with NLoS |
| FMCW radar | `radar` | radar-equation SNR with coherent integration, range/velocity ambiguity, CFAR detections |
| LiDAR | `lidar` | spinning, rosette, MEMS-raster and Lissajous patterns; motion distortion; per-point Doppler |
| Camera | `camera` | pinhole and fisheye lenses, distortion, rolling shutter, motion blur, exposure, sun/weather, thermal mode |
| Event camera | `event_camera` | log-intensity change events with refractory period and background activity |

### IMU: noise from a datasheet

Inertial noise is specified the way datasheets and the IEEE 952 / Allan-variance practice describe it: angle/velocity random walk
(`N`), bias instability (`B`, a flicker 1/f process) and rate random walk (`K`). `IMU.from_allan` converts those figures (in
deg/√h, deg/h, m/s/√h, µg …) to the simulation's units, and `IMU.from_grade` fills them in for a class of sensor.

<!-- docs-test -->
```python
import math
import numpy as np
from aevion.sensors import IMU
from aevion.sensors.allan import allan_deviation, fit_noise
from aevion.sensors.inertial import IMU_GRADES

assert set(IMU_GRADES) == {"consumer", "industrial", "tactical", "navigation"}
imu = IMU.from_grade("tactical", rate=100.0)
assert imu.cfg["gn"] < IMU.from_grade("consumer", rate=100.0).cfg["gn"]        # a better grade is quieter

# the analysis tool works on any rate signal: here white noise with a known density
rng = np.random.default_rng(0)
dt, density = 0.01, 3e-4
x = rng.normal(0.0, density / math.sqrt(dt), (60_000, 2))
taus, adev = allan_deviation(x, dt)
fit = fit_noise(taus, adev, ("N",))
assert abs(fit["N"] - density) < 0.05 * density                                # recovers the angle random walk
```

The 1/f term is generated by a sum of Gauss–Markov processes whose time constants span `bias_tau_range`, calibrated so the Allan
curve is flat at `bias_instability` over the plateau. Beyond the model: `accel_vibration` / `gyro_vibration` add motor-harmonic
vibration at the rotor frequency *as the IMU samples it* (so it aliases like the real thing), and `temperature_c`, `warmup_c`,
`*_bias_tempco` and `gyro_g_sensitivity` add thermal drift and acceleration sensitivity.

> Note: How well the Allan curve is reproduced
>
> White and random-walk slopes are recovered to a few percent. The flicker plateau is a four-process approximation, so it
> ripples by about ±10 % around the specified value, as the tests assert.

### GNSS: pseudoranges, not positions

`gnss` simulates satellites from synthetic Walker constellations (`gps`, `galileo`, `glonass`, `beidou`), computes pseudorange and
Doppler with ionosphere, troposphere, multipath, receiver clock and ephemeris error, drops satellites below the elevation mask or
under the lock threshold, and solves position, velocity and clock by weighted least squares. The output carries the solution, the
fix flag, the number of satellites and DOP, so availability and accuracy follow the geometry: fewer satellites between
buildings, worse HDOP in an urban canyon, no fix without four satellites. Satellites whose line of sight is blocked by a mesh or
primitive are removed, or (with probability `nlos_probability` in `urban` mode) received as a reflected path with a positive
bias.

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation

cfg = {
    "duration": 6.0,
    "world": {"type": "scene", "params": {"origin": {"lat": 47.37, "lon": 8.54, "alt": 400.0}, "utc": "2026-06-21T10:00:00Z"}},
    "trajectory": {"type": "circle", "params": {"center": [0, 0, 20.0], "radius": 10.0, "omega": 0.3}},
    "sensors": ["imu", "mag", "baro", {"type": "gnss", "name": "gps", "params": {"rate": 5.0}, "record": True}],
    "estimator": {"type": "mahony_kf", "handover_s": 2.0},
    "faults": [{"type": "gnss_jamming", "t_start": 3.5, "params": {"pos": [0, 0, 20.0], "power_dbm": 45.0, "sensor": "gps"}}],
}
arch = Simulation(cfg).run()
gps = arch["sensors/gps"]                       # (epochs, vehicles, 11): pos(3) vel(3) fix n_sat hdop pdop cn0
t = arch["sensors/gps/t"] - arch.t[0]
assert gps.shape[2] == 11 and gps[t < 3.0, 0, 7].min() >= 6             # satellites in view before the jammer
assert np.all(gps[t < 3.0, 0, 6] == 1.0)                                # fix held
```

The receiver is checked against the textbook: DOP matches the analytic formula, the least-squares solution recovers a known
position from noiseless pseudoranges, and the combined GPS + Galileo geometry keeps at least eight satellites in view on an open sky. With no fix the
sensor publishes nothing, so the estimator sees a gap (not a stale or zero reading). `gnss_jamming` raises the noise floor through
`interference_fn`, lowering C/N0 until lock is lost.

> Warning: Synthetic almanacs
>
> The constellations are idealised Walker patterns, not broadcast ephemerides: they have the right geometry statistics and orbital
> dynamics, but they will not reproduce the satellites visible at a given real date. There is no carrier-phase / RTK modelling and
> no spoofing receiver-side detection logic beyond the `gnss_spoofing` fault.

### Radar, LiDAR and camera

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation

cfg = {
    "duration": 1.5,
    "trajectory": {"type": "hover", "params": {"pos": [0, 0, 5.0]}},
    "world": {"type": "scene", "params": {"meshes": [{"type": "box", "center": [15, 0, 5], "size": [2, 20, 10]}]}},
    "environment": {"weather": {"preset": "overcast"}},
    "sensors": [
        "imu",
        {"type": "radar", "name": "radar", "params": {"rate": 10.0, "ground_clutter": False}, "record": True},
        {"type": "lidar", "name": "lidar", "params": {"rate": 10.0, "pattern": {"type": "rosette", "points_per_scan": 400}, "doppler": True}, "record": True},
        {"type": "camera", "name": "cam", "params": {"rate": 10.0, "width": 32, "height": 24, "modes": ["depth", "rgb"], "lens": "equidistant", "fov_deg": 120.0}, "record": True},
        {"type": "pitot", "name": "airspeed", "params": {"rate": 20.0}, "record": True},
    ],
}
arch = Simulation(cfg).run()
radar = arch["sensors/radar"]        # (frames, vehicles, beams, 6)
lidar = arch["sensors/lidar"]        # (frames, vehicles, points): range per point of the rosette scan
depth = arch["sensors/cam"]          # (frames, vehicles, rows, columns)
assert radar.shape[3] == 6 and lidar.shape[2] == 400 and depth.shape[2:] == (24, 32)
finite = lidar[np.isfinite(lidar) & (lidar > 0)]
assert finite.size and 12.0 < float(np.median(finite)) < 40.0           # the wall at x = 14 m is what the scan sees
```

FMCW radar. Range resolution is `c / 2B`, velocity resolution `λ / 2NT_c`, and the unambiguous velocity `±λ / 4T_c`; targets beyond
it alias exactly as in a real chirp radar. Detection uses the radar equation with `N` coherently integrated chirps (SNR gain
10 log₁₀ N), a threshold in dB, and optionally a CA-CFAR over the range–Doppler map. Each beam reports range, azimuth,
elevation, radial velocity, SNR and cross-section. Cross-section falls with the incidence angle, and ground clutter is available.
The radar returns the *first surface* along each beam: no multipath, no multiple targets in one resolution cell.

LiDAR. `pattern` selects the beam layout: `spinning` (the original grid), `rosette` (non-repetitive, Livox-style), `mems_raster` and
`lissajous`. `motion_distortion: true` timestamps each point so a moving vehicle skews the scan the way it does in hardware, and
`doppler: true` adds a per-point radial velocity (FMCW LiDAR). Rain and fog extinguish returns using the weather's extinction
coefficient.

Camera. `lens` is `pinhole` (with Brown–Conrady `distortion`) or a fisheye model (`equidistant`, `equisolid`, `stereographic`,
`orthographic`). Rolling shutter staggers the exposure time across rows; `exposure_s` with `blur_samples` blurs along the motion;
`exposure_ev` and `vignetting` shape the image; the sun's elevation (from `environment.utc`) lights the scene, and rain and fog fade
distant pixels. The `thermal` mode returns a temperature image from surface temperature and the weather. The `event_camera` sensor
differences the log intensity of successive renders and emits events with a contrast threshold, a refractory period and background
noise.

> Warning: Approximate imaging
>
> The camera is a ray-cast renderer with simple lighting. It is useful for geometry (depth, optical-flow, event data, lens effects)
> and for exercising a vision pipeline, but it is not photorealistic and has no textures: expect a domain gap for learned
> perception. Non-static LiDAR patterns are not drawn in the viewer, which shows the recorded points.

### Sensor faults

Beyond `gps_outage`, the fault library covers the failure modes that break estimators in practice: `sensor_bias_step`, `sensor_drift`,
`sensor_scale`, `sensor_stuck`, `sensor_dropout`, `sensor_noise_scale`, `sensor_saturation`, `sensor_spikes`, `sensor_nan`,
`sensor_delay` and `sensor_delay_spikes` (late, in-order deliveries), plus `gnss_jamming` and `gnss_spoofing`. Each names a `sensor`,
optionally a `field`, a start (and end) time, and parameters.

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation

cfg = {
    "duration": 8.0,
    "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 3.0, "omega": 0.6}},
    "sensors": ["imu", "gps", "baro", "mag"],
    "estimator": {"type": "mahony_kf", "handover_s": 3.0},
    "faults": [{"type": "sensor_nan", "t_start": 4.0, "params": {"sensor": "gps", "field": "pos", "probability": 0.3}}],
}
arch = Simulation(cfg).run()
assert np.all(np.isfinite(arch["est_pos"]))       # non-finite measurements are rejected, not absorbed by the filter
```

### Limits

* Only the first surface along a ray is seen, by every ray-casting sensor.
* The camera has no textures, indirect light or specular reflection.
* GNSS constellations are synthetic; there is no carrier phase.
* The IMU's 1/f noise is a four-process approximation of flicker.
* Radar clutter is a simple ground model; there is no multipath, micro-Doppler or interference between radars.
* Sensor *mounting* is rigid; flexible structures are not modelled.

---

<!-- page: control -->

## Advanced control

The basic controllers (cascaded PID, geometric SO(3), LQR hover) are joined by five more, all registered under
`controller:` and all running on the same batched, backend-generic code. None is better in every case; each suits a particular
disturbance or constraint, as the table shows.

| Controller | Registered as | Use it when | Main assumption |
| --- | --- | --- | --- |
| Model-predictive control | `mpc` | the reference is aggressive or you need to respect tilt and thrust limits ahead of time; a steady wind or payload must not leave an offset | a double-integrator translational model; a disturbance observer supplies offset-free tracking |
| Sampling-based MPC (MPPI) | `mppi` | the cost is not convex: obstacle avoidance through a signed-distance field | enough samples (the default is 512); the result is stochastic but seeded |
| Incremental NDI | `indi` | an unmodelled torque (centre-of-mass shift, a damaged rotor) leaves the PD attitude loop with an offset | angular acceleration is measured (here: rotor-speed feedback and rotor reaction torque), so no model of the airframe is needed |
| L1 adaptive | `l1` | wind or payload changes the plant while the flight is under way | the uncertainty is matched, slowly varying and bounded (`limit`) |
| Sliding mode | `smc` | the disturbance is bounded and unknown | `k_switch` must exceed the disturbance bound, or the sliding condition fails |

```yaml
controller: {type: mpc, params: {horizon: 20, step: 0.1, observer: true}}
```

### Model-predictive control

The translational loop is a finite-horizon quadratic program over acceleration commands, one per axis (the axes are decoupled
by the double-integrator model), with box constraints on acceleration that come from the tilt limit and the available thrust.
It is solved for all axes and all vehicles at once by a batched ADMM solver (`BatchedBoxQP`) that warm-starts from the shifted
previous solution, so a solve is a handful of matrix–vector products. The attitude loop below it is the geometric controller
that the other controllers share.

A Kalman disturbance observer (`DisturbanceObserver`) augments each axis with a constant-disturbance state estimated from
position and velocity, and the MPC feeds it forward. That is what makes tracking *offset-free*: a 5 m/s wind or a 0.4 kg payload
dropped on the vehicle is estimated and cancelled instead of leaving a steady error. The unit tests compare it with a PD law on a
constant disturbance, and an engine test checks wind plus a payload change against the geometric controller. `observer: false`
turns the estimate off.

> Note: The default horizon is feasible, not magic
>
> The reference must be feasible within the limits over the horizon. A reference that demands more acceleration than the tilt
> limit allows is tracked as closely as the constraint permits, not exactly. The default limit is 40°, in line with the
> geometric controller; MPC does *not* make a quadrotor do what its thrust cannot.

### MPPI

Model-predictive path-integral control samples `samples` noisy acceleration sequences, scores them with the same quadratic cost
plus an obstacle term from a signed-distance field (`sdf`, with `clearance` and `obstacle_weight`), and averages them with
exponential weights. The temperature is *relative* to the cost scale, so it does not need retuning when the weights change.
A handful of samples gives a visibly noisy command; the tests use 256 for multi-vehicle runs and the default is 512.

### INDI

INDI computes the increment of control needed to produce the commanded angular acceleration from the *measured* angular
acceleration, which makes it insensitive to an unmodelled torque. In this simulator the measured quantity is derived from the
measured rotor speeds (and the rotor's reaction torque) rather than from differentiating the gyro, filtered with matching
low-pass filters on the actuator and the measurement so the increments stay consistent. Without the matched filters INDI
diverges, and the tests cover recovery from a tilt (as fast as the geometric loop) and a centre-of-mass shift that leaves the PD
attitude loop with a drift INDI removes.

### L1 adaptive and sliding mode

`l1` runs a state predictor and a low-pass-filtered adaptive law (`predictor_pole`, `omega_c`): the filter bandwidth `omega_c`
decides how fast the disturbance estimate acts and how much high-frequency noise reaches the command. `smc` uses a sliding
surface on the position/velocity error and a boundary-layer switching term to avoid chattering; inside the layer it behaves like
a PD law with high gain.

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation


def mean_error(controller, seed=1):
    cfg = {
        "duration": 14.0,
        "seed": seed,
        "trajectory": {"type": "figure8", "params": {"center": [0, 0, 6.0], "size": 4.0, "period": 14.0}},
        "wind": {"type": "constant", "velocity": [5.0, 2.0, 0.0]},
        "faults": [{"type": "payload_change", "t_start": 3.0, "params": {"mass": 0.4}}],
        "controller": controller,
    }
    run = Simulation(cfg).run()
    err = np.linalg.norm(run["pos"][:, 0] - run["ref_pos"][:, 0], axis=1)
    return float(err[run.t > 2.0].mean())


e_geo, e_l1 = mean_error("geometric"), mean_error("l1")
assert e_geo > 0.05                 # a steady wind plus an unexpected 0.4 kg payload leave the nominal loop with an offset
assert e_l1 < 0.25 * e_geo          # the adaptive law estimates and cancels it
```

> Warning: Sliding mode needs a known bound
>
> With `k_switch` below the disturbance acceleration the tracking error grows several-fold (the test uses 2 against a few
> m/s²). The controller does not tell you the bound is wrong; check the error, or use `l1`/`mpc`, which estimate the
> disturbance instead of bounding it.

### Several vehicles

All of them are batched: `n` vehicles are controlled by one object, and MPC, MPPI and the observer carry per-vehicle state. The
tests fly a three-vehicle formation with MPPI and check the tracking error and finite commands; MPPI also runs on the torch
backend with the same limits. Per-vehicle equality with single-vehicle runs is not asserted.

---

<!-- page: estimation -->

## State estimation

This section covers an error-state Kalman filter (ESKF) that fuses the IMU with GNSS, magnetometer, barometer and, when configured,
rangefinder and optical-flow measurements; consistency checks on the filter's own uncertainty; an observability analysis; and a
log-odds occupancy grid for mapping.

### The error-state filter

`ESKF` keeps a nominal state (position, velocity, attitude quaternion, gyro and accelerometer biases) that is integrated by
strapdown mechanisation, and a 15-dimensional error state with covariance `P` that is propagated and corrected. The attitude
error is expressed in the body frame; the error transition uses the second-order expansion of the rotation and the nominal
attitude is integrated across the step rather than held at its start-of-step value. Corrections use the Joseph form, so `P`
stays symmetric and positive semi-definite, and each measurement is gated with a χ² test on its innovation (`gate`), so one
outlier does not drag the estimate away.

| Feature | Detail |
| --- | --- |
| Measurements | GNSS position and velocity, barometric height, magnetic heading (tilt-compensated Jacobian), rangefinder height, optical-flow velocity |
| Delayed measurements | a snapshot ring buffer: a GNSS fix that is `latency` old is applied at its own time by rewinding to the snapshot, fusing, and replaying the stored IMU samples |
| Gating | χ² gate per measurement with the innovation covariance `S = HPHᵀ + R`; rejected fixes are counted in `n_rejected` |
| Batching | `n` vehicles are filtered at once; every operation is a batched matrix expression |
| Backends | NumPy; the code is written against the backend-generic ops |

Use it from a configuration as an estimator:

```yaml
estimator: {type: eskf, handover_s: 3.0}
sensors: [imu, mag, baro, {type: gps, params: {latency: 0.15}}]
```

The engine records `est_pos`, `est_quat`, `est_vel` and `est_pos_std`, the filter's one-sigma position uncertainty.

> Warning: `est_pos_std` is optimistic against a slowly wandering GNSS error
>
> The filter treats each GNSS fix as white noise. The simulated receiver's error is a Gauss–Markov process with a 60 s time
> constant (as real GNSS errors are), so successive fixes share the same error and the filter's covariance shrinks as if
> they were independent: in the example below the position error stays near the receiver's ~0.8 m wander while
> `est_pos_std` falls to about 0.1 m. This is a modelling choice of the *filter*, not a bug in the statistics: the Monte-Carlo
> consistency tests below use white measurement noise, where the ratios are 1.0. If the covariance matters downstream, inflate
> `gnss_pos_std` to the wander amplitude, or add a GNSS-bias state.

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation

sim = Simulation({
    "duration": 12.0,
    "seed": 1,
    "trajectory": {"type": "circle", "params": {"center": [0, 0, 6.0], "radius": 4.0, "omega": 0.6}},
    "sensors": ["imu", "mag", "baro", {"type": "gps", "params": {"latency": 0.15}}],
    "estimator": {"type": "eskf", "handover_s": 3.0},
})
run = sim.run()
late = run.t > 4.0
err = np.linalg.norm(run["est_pos"][:, 0] - run["pos"][:, 0], axis=1)
assert err[late].mean() < 1.5                                   # metre-level with consumer-grade sensors
assert run["est_pos_std"].shape == (len(run.t), 1, 3)           # the filter reports its own uncertainty every step
assert np.all(np.isfinite(run["est_pos_std"]))
```

### Is the filter consistent?

A Kalman filter is only useful if its covariance is the truth. `aevion.estimation.consistency` implements the standard checks
(Bar-Shalom, Li and Kirubarajan, *Estimation with Applications to Tracking and Navigation*, §5.4):

* NEES, the normalised estimation error squared `eᵀ P⁻¹ e`, which should average to the state dimension;
* NIS, the same for the innovations, which should average to the measurement dimension;
* `nees_interval(dof, n_runs)` gives the two-sided χ² acceptance interval for the average over Monte-Carlo runs, and
  `consistency_report` / `is_consistent` apply it.

The test-suite runs Monte-Carlo ensembles of the filter with known noise and checks the average NEES/NIS ratios sit at about
1.0 inside that interval, for the attitude, velocity and position blocks separately.

<!-- docs-test -->
```python
from aevion.estimation import nees_interval

lo, hi = nees_interval(dof=3, n_runs=50)          # the band for an average over 50 runs of a 3-dof error
assert lo < 3.0 < hi
lo1, hi1 = nees_interval(dof=3, n_runs=1)
assert (hi - lo) < 0.2 * (hi1 - lo1)              # averaging 50 runs narrows the band about sixfold
```

> Warning: What this does and does not show
>
> Consistency on simulated data proves that the implementation matches its own noise model. A real receiver has correlated,
> non-Gaussian errors the simulation only approximates (multipath, ionospheric gradients), so re-tune `R` against real
> logs before trusting the covariance on hardware.

### Observability

`observability_matrix` builds the discrete observability matrix of a linearised system, and `observable_rank`,
`unobservable_subspace` and `direction_observability` report which state directions a given motion and sensor set can see.
The tests use them to show the classical results: with GNSS and magnetometer, heading is observable only with a magnetometer
or while accelerating; accelerometer bias is not observable in hover; and a stationary vehicle cannot observe its own heading
from an IMU and GNSS alone.

### Occupancy mapping

`OccupancyGrid` is a log-odds voxel map over an axis-aligned box. `integrate` (or `insert_ranges` / `insert_points` for sensor
data) traces each ray with a fast voxel walk, adds negative evidence along the free space and positive evidence at the hit, and
clamps the stored log-odds so the map stays adaptable (the defaults are the OctoMap values). The grid is bounded by
`MAX_VOXELS`, so a mistyped resolution fails with a clear message instead of exhausting memory. `to_sdf` returns a signed
distance field for the avoidance filters, and `PlanningSpace.from_occupancy(grid, inflation)` turns the map into a planning
space, so a map made from a LiDAR in flight can be planned through.

<!-- docs-test -->
```python
import numpy as np
from aevion.estimation.mapping import OccupancyGrid

grid = OccupancyGrid((-5, -5, 0, 5, 5, 4), resolution=0.5)
origin = np.array([[0.0, 0.0, 1.0]] * 2)
for _ in range(3):                                  # a wall seen repeatedly at x = 3
    grid.integrate(origin, np.array([[3.0, 0.0, 1.0], [3.0, 1.0, 1.0]]))
assert grid.occupied([[3.0, 0.0, 1.0]])[0] and not grid.occupied([[1.5, 0.0, 1.0]])[0]
assert grid.probability([[1.5, 0.0, 1.0]])[0] < 0.4                    # the rays passed through, so the cells before the wall are free
assert grid.distance([[1.0, 0.0, 1.0]])[0] > 1.0                        # about 2 m from the wall
```

---

<!-- page: planning -->

## Planning and avoidance

The planning stack sits between the goal and the controller: a voxel planning space built from the
world, graph and sampling path planners, trajectory generation that turns a path into a smooth, dynamically feasible
reference, collision avoidance for moving obstacles and other vehicles, and mission logic as behaviour trees.

```text
world / occupancy map ──► PlanningSpace ──► path (A*, Theta*, D* Lite, RRT*, PRM)
                                                │
                                                ▼
                       min-snap or time-optimal trajectory ──► controller ──► vehicle
                                                ▲
              ORCA-3D / control-barrier filters ┘  (adjust the reference around moving obstacles)
```

### The planning space

`PlanningSpace` is a boolean voxel grid. `PlanningSpace.from_world(world, resolution, inflation)` rasterises primitives, meshes
and terrain through the signed-distance field, and inflates the obstacles by the vehicle radius plus a margin, so a path
that stays in free voxels keeps that clearance. `from_occupancy` builds the same thing from a mapped
[`OccupancyGrid`](#occupancy-mapping). Segment queries use an exact voxel walk (Amanatides–Woo DDA, compiled with
numba) rather than sampling along the segment, so a thin wall cannot be stepped over.

### Path planners

| Planner | Function | Properties |
| --- | --- | --- |
| A\* | `astar` | optimal on the 6-, 18- or 26-connected grid; `weight > 1` gives weighted A\* (bounded-suboptimal, faster) |
| Theta\* | `theta_star` | any-angle: parents are re-chosen along line of sight, so paths are not constrained to grid directions; the default |
| D\* Lite | `DStarLite` | incremental replanning: when the map changes it repairs the previous search instead of starting over |
| RRT\* | `rrt_star` | asymptotically optimal sampling planner with informed sampling once a first path exists; for cluttered 3-D spaces |
| PRM | `PRM` | builds a roadmap once and answers many queries |

`plan_path(space, start, goal, method="theta_star")` dispatches to any of them and returns a `PlanResult` (`path`, `cost`,
`success`, `expansions`, `time_s`, `reason`). A failed plan is never an exception at this level: `success` is `False` and
`reason` says why (start or goal inside an obstacle, no route, or the expansion budget). The tests check optimality against
Dijkstra on random grids, that Theta\* is collision-free and shorter than the grid path, and that D\* Lite's repaired path matches a
fresh search after every map change.

<!-- docs-test -->
```python
import numpy as np
from aevion.planning import PlanningSpace, plan_path, path_length

occ = np.zeros((40, 40, 10), bool)
occ[20, 5:40, :] = True                                  # a wall across the field with a gap at y < 5
space = PlanningSpace(occ, lo=(0.0, 0.0, 0.0), resolution=1.0)
start, goal = (2.0, 20.0, 3.0), (38.0, 20.0, 3.0)

astar_path = plan_path(space, start, goal, method="astar")
theta = plan_path(space, start, goal, method="theta_star")
assert astar_path.success and theta.success
assert space.path_free(theta.path)                        # the exact segment test agrees
assert theta.cost <= astar_path.cost + 1e-9               # any-angle is never longer than the grid path
assert path_length(theta.path) > 40.0                     # it has to go round the wall, through the gap

blocked = np.zeros((20, 20, 5), bool); blocked[10, :, :] = True
res = plan_path(PlanningSpace(blocked, (0, 0, 0), 1.0), (2, 10, 2), (18, 10, 2), method="astar")
assert not res.success and res.reason                     # a sealed wall: a reported failure, not an exception
```

### From a path to a trajectory

A grid path is not flyable. `plan_trajectory(space, path, v_max, a_max)` does three things:

1. prunes and shortcuts the path (line-of-sight pruning, then random shortcutting), optionally resamples it;
2. fits a minimum-snap polynomial through the waypoints by solving the KKT system of the constrained quadratic program
   (Mellinger and Kumar 2011; Richter, Bry and Roy 2013), with times from a trapezoidal heuristic;
3. checks the result against the space and, when the smooth curve clips an obstacle, inserts a waypoint on the violated
   segment and re-fits (`max_repairs` bounds the loop); times are then stretched until the speed and acceleration limits hold.

`time_optimal_profile` is the alternative when the geometric path is fixed: it computes the fastest speed profile along the
curve under speed, acceleration and (with `ArcCurve`) curvature limits with the forward–backward pass. The returned
`PlannedTrajectory` is a normal `Trajectory`, so every controller and the viewer take it.

From a configuration, the `planned` trajectory does the search, the fit and the repair when the simulation is built, and says
so if there is no route:

```yaml
world: {type: scene, params: {boxes: [{center: [15, 0, 6], size: [4, 14, 12]}], bounds: [-5, -15, 0, 35, 15, 20]}}
trajectory:
  type: planned
  params: {start: [0, 0, 3], goal: [30, 0, 3], v_max: 4.0, a_max: 3.0, inflation: 0.8, resolution: 1.0}
```

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation

sim = Simulation({
    "duration": 22.0,
    "world": {"type": "scene", "params": {"boxes": [{"center": [15, 0, 6], "size": [4, 14, 12]}], "bounds": [-5, -15, 0, 35, 15, 20]}},
    "trajectory": {"type": "planned", "params": {"start": [0, 0, 3], "goal": [30, 0, 3], "v_max": 4.0, "a_max": 3.0, "inflation": 0.8, "resolution": 1.0}},
})
run = sim.run()
assert sim.traj.info["peak_speed"] <= 4.0 * 1.02                       # the limits are honoured by construction
assert np.linalg.norm(run["pos"][-1, 0] - [30, 0, 3]) < 0.3            # the vehicle flew around the building to the goal
assert not any(e["type"] == "collision" for e in run.meta["events"])
```

### Avoiding moving obstacles and other vehicles

Planned paths assume a static world. Two reactive layers handle what the plan did not know about:

* `ORCA3D` (Optimal Reciprocal Collision Avoidance, van den Berg et al. 2011, extended to three dimensions): each agent
  gets a half-space of velocities that keep it collision-free for `time_horizon`, assuming its neighbour takes half the
  responsibility, and chooses the velocity closest to its preferred one inside the intersection. Static obstacles enter through
  a signed-distance field. The intersection of half-spaces is solved with a projection routine (Hildreth's method with
  Dykstra's correction for the ball constraint), which converges to the exact projection rather than stopping at the first
  feasible point.
* `CBFSafetyFilter`: a control-barrier-function filter that minimally modifies a commanded velocity or acceleration so that
  `h(x) = SDF(x) − radius − margin` stays non-negative with the class-K rate `alpha`, with optional speed and acceleration
  bounds. It is a safety *filter*: it only intervenes when the barrier condition would be violated.

> Note: What the guarantees are
>
> ORCA's collision-free property holds when every agent follows ORCA and the model (velocity-controlled disks/spheres, perfect
> state) is true; the CBF's holds for the single-integrator or double-integrator model it is built on. A real vehicle tracks
> commands with lag, so both take a `margin` to cover tracking error. The tests simulate the agents (point-mass kinematics, ten
> agents swapping places across a sphere, moving and static obstacles) and measure the minimum distance reached; they do not
> cover full rotor-level dynamics.

The constrained projections they rely on, `project` (halfspaces and a ball) and `solve_qp` (an OSQP-style ADMM solver with
adaptive `rho`, warm starts and infeasibility detection), are public and usable directly.

### Missions as behaviour trees

`BehaviorTreeMission` runs a behaviour tree whose leaves fly the vehicle: `Takeoff`, `GoTo` (plans a path through the space),
`Hover`, `Land`, `Wait`, and conditions `BatteryAbove` and `AltitudeAbove`. Composites are `Sequence`, `Fallback` and
`Parallel`; decorators are `Inverter`, `ForceSuccess`, `ForceFailure`, `Repeat`, `Retry` and `Timeout`. A tree can be written in
Python or as a dictionary / YAML (`tree_from_dict`). Because it is a trajectory, a mission runs in the ordinary engine, sees the
vehicle's battery and position, and can end the run when it finishes (`on_finish: stop`).

<!-- docs-test -->
```python
from aevion.engine import Simulation

tree = {
    "type": "sequence",
    "children": [
        {"type": "takeoff", "altitude": 4},
        {"type": "goto", "waypoint": [10, 0, 4]},
        {"type": "land"},
    ],
}
sim = Simulation({
    "duration": 30.0,
    "world": {"type": "scene", "params": {"boxes": [], "bounds": [-5, -15, 0, 35, 15, 20]}},
    "trajectory": {"type": "mission", "params": {"tree": tree, "home": [0, 0, 0.05], "v_max": 3.0, "resolution": 1.0}},
})
run = sim.run()
assert run["pos"][:, 0, 2].max() > 3.8                     # it climbed,
assert abs(run["pos"][-1, 0, 0] - 10.0) < 0.5              # went to the waypoint,
assert run["pos"][-1, 0, 2] < 0.3                          # and landed
```

A guard such as `Fallback([Sequence([BatteryAbove(0.3), mission]), Land()])` diverts a mission when the battery runs low; the test
suite does this with a nearly empty pack and checks that the mission is cut short.

### Limits

* Planning is on a voxel grid: resolution trades accuracy (thin obstacles smaller than a voxel are inflated to a voxel) against
  speed and memory. `MAX_VOXELS` bounds the grid.
* The trajectory check samples the polynomial at `check_dt`; it is a dense check, not a certificate. Keep `margin` above the
  controller's tracking error.
* The planners treat the vehicle as a sphere. Anisotropic shapes need the inflation of their circumscribing sphere.
* Mission trees tick at the control rate and are not real-time-scheduled: `Timeout` is in simulated seconds.

---

<!-- page: wireless -->

## Wireless

Aevion's wireless package models the radio link of a UAV from the antenna to the application, in layers that can be used on
their own or through one integrated network model driven by the simulation engine. It is built for UAV communication research:
cellular-connected UAVs, UAV base stations and relays, swarm ad-hoc networks, satellite links, and RF threats to navigation and
control. Every model states where it is valid, and `tests/test_wireless_*.py` checks each against an analytic result, a
hand-computed reference or a published number.

> Warning: What this is, and is not
>
> These are system-level abstractions (link budgets, statistical channels, an interval-level NR MAC/PHY), the right
> fidelity for mobility, coverage, handover, scheduling and resilience studies over minutes of flight. They are *not* a
> waveform-level link simulator, and the ray tracer is first-order specular with no diffuse scattering. Coefficients of the
> 3GPP models were transcribed from the published tables; verify them against the standard before quoting absolute numbers in a
> paper (the test-suite proves the implementation matches the formulas as written, not that the formulas match the standard).

### Map of the package

| Subpackage | What it provides |
| --- | --- |
| `propagation` | Free space, two-ray, log-distance, Al-Hourani, 3GPP TR 38.901 (UMa, UMi, RMa, InH), TR 36.777 aerial models, ITU-R P.526 knife-edge and Bullington diffraction, image-method ray tracing with Fresnel reflection |
| `fading` | Rayleigh, Rician, Nakagami; time-correlated Jakes process driven by the UAV's own Doppler; Gudmundson spatially correlated shadowing; LCR/AFD/coherence time |
| `antennas` | Isotropic, dipoles, 3GPP element and sector patterns, Gaussian beam, parabolic; ULA/UPA, DFT codebooks, MRT/ZF/MMSE, beam management; attitude-dependent UAV antenna gain |
| `channel` | Air-to-ground channel, MIMO (geometric + Kronecker correlation, capacity, water-filling), blockage (Boolean model, Markov, self-blockage), delay spread, link kinematics |
| `phy` | NR numerology and PRB tables, MCS/CQI tables, attenuated-Shannon BLER model, link adaptation with OLLA, HARQ with chase combining, EESM |
| `mac_network` | Schedulers (RR, max C/I, PF), A3 handover, radio-link failure, relay, mesh routing, Bianchi CSMA, and the integrated `nr_cellular` network |
| `ris` | Reconfigurable intelligent surface: spherical-wave cascaded channel, phase optimisation and quantisation |
| `ntn` | LEO/MEO/GEO geometry, pass Doppler, link budget, gaseous and rain attenuation |
| `maps` | Radio environment maps, coverage radius and optimal UAV-BS altitude, map reconstruction (IDW, RBF, kriging) |
| `security` | Jamming and GNSS spoofing threat models, engine faults `gnss_jamming`, `gnss_spoofing`, `network_jamming` |
| `metrics` | Outage, capacity (incl. finite blocklength), age of information, latency, run-level link metrics |
| `interop` | ns-3 mobility/link traces, OBJ geometry for external ray tracers |

The original names (`AirToGroundChannel`, `CellularNetwork`, ...) are kept, and the results they produce are
reproduced bit for bit.

### Propagation

All models implement `PathLossModel`: `path_loss_db(geometry, los)`, `los_probability(geometry)`, `shadowing_std_db(geometry, los)`.
Randomness is applied by the caller (`realise()`), so models are deterministic and testable.

<!-- docs-test -->
```python
import numpy as np
from aevion.wireless.propagation import LinkGeometry, TR38901, TR36777, make_model, realise

g = LinkGeometry.from_distances(d2d=[100.0, 500.0, 2000.0], h_bs=25.0, h_ut=1.5)
uma = TR38901(3.5e9, "uma")
print("UMa LoS  :", uma.path_loss_db(g, True).round(1), "dB")
print("UMa NLoS :", uma.path_loss_db(g, False).round(1), "dB")
print("P(LoS)   :", uma.los_probability(g).round(3))

# a drone flies much higher than a handset: TR 36.777 changes both the loss and the LoS probability
drone = LinkGeometry.from_distances([500.0], 25.0, 100.0)
print("UMa-AV P(LoS) at 100 m:", TR36777(2e9, "uma_av").los_probability(drone).round(2))

pl, los = realise(make_model("tr38901", freq_hz=3.5e9, scenario="umi"), g, np.random.default_rng(0))
assert pl.shape == (3,) and los.dtype == bool
```

Validity. `model.validity(geometry)` reports links outside the specified distance, frequency and terminal-height range
(UMa/UMi 10 m-5 km and UT height 1.5-22.5 m, RMa to 10 km and 10 m, InH 1-150 m; TR 36.777 up to 300 m height). Values are still
returned, but treat them as extrapolation. Use the aerial models for UAVs: the TR 38.901 UMa NLoS formula contains
`-0.6 (h_UT - 1.5)`, which silently removes about 29 dB of loss at 50 m. `NrCellularNetwork` therefore defaults to `tr36777`
(which uses TR 38.901 below 22.5 m) and warns once if a terrestrial model is applied to a UAV outside its height range.

#### Diffraction and reflections from the actual world

For links in a `World` with buildings:

* `world_diffraction_loss_db` computes ITU-R P.526 knife-edge loss with the Bullington equivalent edge over building roofs
  and terrain profile. Bullington replaces all obstacles by one edge and tends to under-estimate loss behind many obstacles.
* `trace_paths` finds the direct path, the ground reflection and first-order reflections from every box face by the image
  method, each with its Fresnel coefficient (ITU-R P.2040 material constants). The coherent sum gives the multipath gain; the
  power-delay profile gives delay spread; the direct-to-reflected power ratio gives a geometric Rician K-factor.

<!-- docs-test -->
```python
import numpy as np
from aevion.environments.world import World
from aevion.wireless.propagation import trace_paths, world_diffraction_loss_db

city = World()
city.add_box_corners([25, -10, 0], [35, 10, 20])           # a building between the base station and the UAV
tx, rx = np.array([[0.0, 0.0, 25.0]]), np.array([[60.0, 0.0, 15.0]])
print("diffraction over the roof:", world_diffraction_loss_db(city, tx, rx, 2.4e9).round(1), "dB")
paths = trace_paths(city, tx, rx, 2.4e9)
print("paths:", paths.labels, "valid:", paths.valid[0].astype(int))
```

### Fading and shadowing

`JakesProcess` is a sum of phasors with independent phases and evenly spaced arrival angles: its envelope is Rayleigh and its
autocorrelation follows Bessel `J0(2 pi f_d tau)`; phases integrate the *instantaneous* Doppler, so a UAV that accelerates
changes its fading rate consistently. Tests verify unit power, KS-consistency with the exponential distribution and the
Bessel autocorrelation. `CorrelatedShadowing` is a first-order autoregressive field whose correlation is
`exp(-distance / d_corr)` with the TR 38.901 decorrelation distances.

Analytic companions are provided so simulations can be checked: Jakes spectrum, level-crossing rate, average fade duration
(`LCR x AFD = P_outage` is asserted), exact coherence time, Rician/Nakagami moment matching.

### Antennas, beams and the UAV's attitude

A UAV pitches and rolls to fly, and its antennas rotate with it. `UavAntenna` evaluates a pattern in the antenna frame
reached from the world frame through the vehicle quaternion and the mounting rotation, so the link budget responds to
manoeuvres: a downward patch loses `12 (tilt / 65 deg)^2` dB when the UAV tilts, a vertical dipole has a null overhead.
`BeamManager` models NR-style beam sweeping: between sweeps the beam is fixed, so a manoeuvring UAV drifts out of it.

<!-- docs-test -->
```python
import math
from aevion.wireless.antennas import UavAntenna, UniformLinearArray, dft_codebook

patch = UavAntenna.preset("patch_down")
level = float(patch.gain_dbi([1, 0, 0, 0], [0, 0, -1]))
q30 = [math.cos(math.radians(15)), 0, math.sin(math.radians(15)), 0]       # 30 degrees of pitch
print(f"downward patch: {level:.1f} dBi level, {float(patch.gain_dbi(q30, [0, 0, -1])):.1f} dBi at 30 deg pitch")

array = UniformLinearArray(8)
codebook = dft_codebook(array, oversample=2)
print(len(codebook), "beams; half-power beamwidth", round(array.half_power_beamwidth_deg(), 1), "deg")
```

### The integrated network: `nr_cellular`

`NrCellularNetwork` (config `network: {type: nr_cellular, ...}`) evaluates, once per network interval and for every UAV:

1. median path loss of the chosen model, LoS state (probabilistic with spatial persistence, or geometric by ray casting),
   spatially correlated shadowing, optional blockage and diffraction;
2. cell sector gain towards the UAV and the UAV antenna gain from its actual attitude;
3. time-correlated fading from the UAV's own Doppler;
4. layer-3-filtered RSRP driving A3 handover (hysteresis, time-to-trigger, interruption, failures, ping-pong) and
   radio-link-failure timers (`T310`, `N310/N311`, recovery);
5. interference from other cells at a given load, plus jammers;
6. a scheduler share against background users, then CQI/MCS with OLLA, BLER and HARQ to goodput and latency.

<!-- docs-test -->
```python
from aevion.engine import Simulation

config = {
    "duration": 8.0, "seed": 1,
    "trajectory": {"type": "circle", "params": {"center": [300, 0, 60.0], "radius": 120.0, "omega": 0.25}},
    "network": {
        "type": "nr_cellular", "rate_hz": 10,
        "params": {
            "sites": [{"pos": [0, 0, 25], "sectors": 1}, {"pos": [600, 0, 25], "sectors": 1}],
            "propagation": {"model": "tr36777", "scenario": "uma_av"},
            "freq_hz": 3.5e9, "bandwidth_mhz": 100, "scs_khz": 30,
            "background_users": 3, "scheduler": "proportional_fair",
        },
    },
}
archive = Simulation(config).run()
print([k for k in archive.channels if k.startswith("link_")][:8], "...")

from aevion.metrics import flight_metrics
m = flight_metrics(archive)
print({k: float(v[0].round(2)) for k, v in m.items() if k.startswith("link_")})
```

Recorded channels: `link_sinr_db`, `link_rate_mbps`, `link_serving`, `link_outage`, `link_rsrp_dbm`, `link_mcs`, `link_bler`,
`link_share`, `link_doppler_hz`, `link_fading_db`, `link_path_loss_db`, `link_n_handovers`, `link_rlf`, ... The report adds a
*link stack* figure, an SINR CDF, and "mobility management" insights automatically.

Abstractions to know about. One call evaluates an *interval* (`rate_hz`), not a slot: ACK/NACK counts are binomial draws
and goodput/latency are the analytic expectations for those slots. The BLER curve is an attenuated-Shannon-based abstraction
(`sigma` about 0.7 dB), not a measured link-level curve. There is no CSI-RS/PMI feedback loop, no channel-estimation error and no
inter-cell scheduling coordination; interference is full-buffer at the configured `load`. The Numba/torch kernels do not apply
here: the network evaluates on NumPy at the network rate.

### Communication-aware learning

The `connected_navigation` task (`aevion.envs.ConnectedNavigationTask`) turns the network into a learning problem: fly to a goal
while staying connected. The agent observes six link features (normalised SINR and rate, unit vector and distance to the serving
cell), and the reward adds a link-quality term and an outage penalty to the progress reward. With a tower between the base
station and the goal the best path is not the straight one. The network is evaluated statelessly with
`NrCellularNetwork.snapshot(positions)` (large-scale power with geometric LoS, optional shadowing, interference, a Shannon rate),
so it scales to thousands of parallel environments.

<!-- docs-test -->
```python
import numpy as np
from aevion.envs import ConnectedNavigationTask, EnvConfig, VecQuadEnv
from aevion.environments.world import World

world = World(bounds=(-250, -150, 0, 250, 150, 150))
world.add_box_corners([-20, -60, 0], [20, 60, 90])                       # a tower between the cell and the goal
task = ConnectedNavigationTask(network={"sites": [{"pos": [-100, 0, 25], "sectors": 1}], "los_mode": "geometric"})
env = VecQuadEnv(8, EnvConfig(task=task, **ConnectedNavigationTask.env_kwargs()), world=world)
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(np.zeros((8, env.act_dim)))
print(obs.shape, "- last six features are the link:", obs[0, -6:].round(2))
```

### Coverage, REMs and where to put a UAV base station

<!-- docs-test -->
```python
from aevion.wireless.mac_network import NrCellularNetwork
from aevion.wireless.maps import optimal_altitude, radio_environment_map

net = NrCellularNetwork(sites=[{"pos": [0, 0, 25], "sectors": 1}, {"pos": [600, 0, 25], "sectors": 1}], seed=0)
rem = radio_environment_map(net, altitude=60.0, extent=(-200, 800, -200, 200), resolution=25.0)
print("area above the outage threshold:", round(rem.coverage_fraction(-6.0), 2), "| cell border length (m):", rem.handover_boundary_length(25.0))

h, r = optimal_altitude(max_path_loss_db=100.0, freq_hz=2e9, env="urban")   # Al-Hourani et al. 2014
print(f"best UAV-BS altitude {h:.0f} m, coverage radius {r:.0f} m")
```

`fig_rem` draws the map with cell sites, borders and a flight path; `reconstruct` rebuilds a map from sparse UAV measurements
(IDW, RBF, kriging with uncertainty).

### Reconfigurable intelligent surfaces

`ris_received_power_dbm` uses the exact spherical-wave cascaded channel (near and far field): the coherent sum of per-element
contributions gives the textbook `N^2` scaling (+12 dB for 4x the elements, asserted), `b`-bit phase quantisation costs
`20 log10(sinc(pi / 2^b))` (0.91 dB at 2 bits), and an unconfigured surface is tens of dB worse than a configured one.

### Satellite links

`SatelliteLink` combines orbit geometry (slant range, a full pass with elevation, range rate and Doppler), free-space loss,
gaseous and rain attenuation (ITU-R P.838 coefficients, 4-40 GHz) and a G/T-based C/N0. A 550 km LEO pass above 10 degrees lasts
about 476 s with +/-46 kHz Doppler at 2 GHz.

### RF threats and resilience

* `gnss_jamming` fault: an interferer degrades the effective C/N0 by Betz' formula; position noise grows and the receiver
  freezes at its last fix when it loses lock, then recovers.
* `gnss_spoofing` fault: threat model with a capture condition (power advantage over the authentic signal) and a position pull
  of configurable rate and limit.
* `network_jamming` fault and `jammers:` in the network: interference added at the UAV, with optional time windows.

These are link-budget threat models for evaluating detection, estimation and control strategies, not signal-level attacks.

### Metrics

`outage_events` (count, durations), `finite_blocklength_rate` (the price of short URLLC packets), `aoi_mm1` / `aoi_periodic_lossy`
(closed-form age of information, checked against simulation), `air_interface_latency_s`, `mm1_sojourn_s`, Jain fairness, and
`aoi_from_archive` for telemetry over a recorded link. `flight_metrics(archive)` includes the link metrics when a network is present.

### Interoperability

`export_ns2_mobility` writes an ns-2 mobility trace that ns-3's `Ns2MobilityHelper` reads (full 3-D), `export_waypoints_csv` and
`export_link_trace_csv` write CSVs, and `world_to_obj` writes the world as a Wavefront OBJ (named groups per object type) for
external ray tracers and DCC tools. Aevion does not import ns-3 or Sionna; direct bridges are on the roadmap.

### Validation summary

| Component | Checked against |
| --- | --- |
| FSPL, 38.901 UMa/UMi/InH path loss, LoS probabilities, TR 36.777 | hand-computed values from the formulas; monotonicity and ordering properties for every model |
| Knife-edge `J(nu)` | ITU-R P.526 values (6.0 dB at grazing, 13.9 dB at `nu = 1`, 20.5 dB at 2.4) |
| Image-method two-ray | exact analytic two-ray interference pattern (perfect conductor), 40 dB/decade far field |
| Rayleigh/Rician/Nakagami | KS tests, analytic variances and outage CDFs (Marcum Q) |
| Jakes fading | unit power, KS against the exponential, Bessel `J0` autocorrelation |
| Shadowing | AR(1) autocorrelation `exp(-d / d_corr)` |
| LCR / AFD | identity `LCR x AFD = 1 - exp(-rho^2)` |
| ULA | array gain `10 log10 N`, 0.886 `lambda / (N d)` beamwidth |
| MIMO | SISO ergodic capacity against `e^(1/g) E1(1/g) / ln 2`; LoS channels nearly rank one |
| NR numerology | PRB tables (TS 38.101), 2.337 Gbit/s peak rate example (TS 38.306) |
| MCS/CQI tables | table size, endpoints and monotonicity (TS 38.214) |
| Link adaptation | OLLA holds the average BLER at the 10 % target and removes a 3 dB reporting bias |
| Handover / RLF | A3 timing arithmetic, TTT/hysteresis/filter effects, ping-pong and failure classification, T310 logic |
| Bianchi CSMA | the fixed-point equations, `n = 1` closed form, throughput trends |
| Age of information | M/M/1 closed form against Monte Carlo; periodic-lossy closed form |
| Finite blocklength | limits to Shannon capacity as `n -> inf` |
| RIS | `N^2` scaling, quantisation loss, back-side blocking |
| Satellite pass | 550 km: 476 s above 10 degrees, +/-46 kHz Doppler, period 95.5 min |
| Optimal UAV-BS altitude | interior optimum of a few hundred metres (Al-Hourani et al. 2014) |
| Jamming | Betz C/N0 formula; free-space 20 dB per decade |

### References

* 3GPP TR 38.901 (channel model), TR 36.777 (UAV channel), TR 36.814 (antenna), TS 38.211/38.214/38.101/38.306 (NR).
* A. Al-Hourani, S. Kandeepan, S. Lardner, "Optimal LAP altitude for maximum coverage", IEEE WCL, 2014.
* ITU-R P.526 (diffraction), P.838 (rain), P.2040 (building materials), P.618 (Earth-space links).
* G. Bianchi, "Performance analysis of the IEEE 802.11 DCF", IEEE JSAC, 2000.
* S. Kaul, R. Yates, M. Gruteser, "Real-time status: how often should one update?", INFOCOM 2012.
* Y. Polyanskiy, H. V. Poor, S. Verdu, "Channel coding rate in the finite blocklength regime", IEEE Trans. IT, 2010.
* W. Tang et al., "Wireless communications with RIS: path loss modelling and experimental measurement", IEEE TWC, 2021.
* R. Betz, "Effect of narrowband interference on GPS code tracking accuracy", ION NTM, 2000 (effective C/N0 under interference).

---

<!-- page: multiagent -->

## Multi-agent coordination

This section covers swarm and multi-UAV algorithms, each checked against a closed form, an independent solver or brute-force
enumeration, and the code that runs them through the simulation's lossy inter-agent link and in a cooperative learning environment.

| Module | What it provides |
| --- | --- |
| `aevion.multiagent.graphs` | communication graphs, Laplacian spectra, algebraic connectivity, consensus weight matrices |
| `aevion.multiagent.consensus` | average, delayed and lossy consensus; push-sum; displacement-based formation control |
| `aevion.multiagent.allocation` | Hungarian, auction (ε-scaling) and CBBA task allocation |
| `aevion.multiagent.coverage` | Voronoi coverage control (Lloyd's algorithm) with density and masks |
| `aevion.multiagent.utm` | 4-D conflict detection, strategic ground delays, prioritised space-time planning |
| `aevion.multiagent.laws` | swarm laws that act on the references through the lossy link: `formation_keeping`, `rendezvous`, `coverage` |
| `aevion.envs.swarm_env` | cooperative multi-agent environment (`VecSwarmEnv`) and a PettingZoo-style `ParallelSwarmEnv` |

### Graphs and consensus

A communication graph is an adjacency matrix (`A[i, j] = 1` when agent `i` hears agent `j`). The Laplacian `L = D − A` governs
linear consensus `x⁺ = (I − εL) x`: on a connected undirected graph with `0 < ε < 2/λ_max` every agent converges to the
mean of the initial values, and the error contracts by `max_{i≥2} |1 − ε λ_i|` per step. `consensus_rate` computes that
factor from the spectrum, and the tests check that the simulated decay matches it.

<!-- docs-test -->
```python
import numpy as np
from aevion.multiagent import adjacency, algebraic_connectivity, average_consensus, consensus_rate, lossy_consensus
from aevion.multiagent.consensus import disagreement

ring = adjacency("ring", 8)
assert abs(algebraic_connectivity(ring) - (2 - 2 * np.cos(2 * np.pi / 8))) < 1e-12     # λ2 of a ring is 2 − 2 cos(2π/n)

x0 = np.random.default_rng(0).normal(size=8)
traj = average_consensus(x0, ring, steps=150)
assert np.allclose(traj.mean(axis=1), x0.mean())                                       # the mean is conserved exactly
assert disagreement(traj[-1]) < 1e-3 and consensus_rate(ring) < 1.0

delayed = lossy_consensus(x0, ring, steps=1500, delay=3)                               # three steps of latency on every link
assert np.allclose(delayed.mean(axis=1), x0.mean()) and disagreement(delayed[-1]) < 1e-6
dropped = lossy_consensus(x0, ring, steps=1500, loss=0.3, seed=1)                      # 30 % of the messages are lost
assert disagreement(dropped[-1]) < 1e-3 and abs(dropped[-1].mean() - x0.mean()) > 1e-6  # they still agree, but no longer exactly on the mean
```

A pure delay keeps the mean exactly; packet loss makes the links asymmetric, so the agents agree on a value *near* the
average (the tests quantify it). `push_sum` reaches the exact average on a strongly connected directed graph, where plain
consensus is not mean-preserving. `formation_step` is consensus on the shifted positions `p_i − d_i`
(Ren and Beard 2008): the formation shape `d` is reached and the centroid kept.

### Task allocation

`hungarian` solves the linear assignment problem exactly (compared with `scipy.optimize.linear_sum_assignment` in the tests,
rectangular and with forbidden pairs). `auction` is Bertsekas' forward auction with ε-scaling: it ends within `m·ε` of the optimum
(`m` = number of objects) and is exact for integer benefits at the default ε; it is the form that distributes naturally.

`cbba` is the Consensus-Based Bundle Algorithm (Choi, Brunet and How 2009) for allocations where an agent's score for a
task depends on the *other* tasks it already holds (travel time, ordering). Each agent builds a bundle greedily from marginal
scores, then agents exchange winning bids over the communication graph and resolve conflicts with the paper's rule table.
Assignments are conflict-free when the graph is connected, and for diminishing-marginal-gain scores (such as the time-discounted
`TravelScore`) the result is at least 50 % of the optimum.

<!-- docs-test -->
```python
import numpy as np
from aevion.multiagent import TravelScore, adjacency, auction, brute_force_allocation, cbba, hungarian
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(1)
cost = rng.uniform(0, 10, (6, 6))
rows, cols, total = hungarian(cost)
assert abs(total - cost[linear_sum_assignment(cost)].sum()) < 1e-9

benefit = rng.uniform(0, 10, (5, 5))
res = auction(benefit, eps=0.02)
assert hungarian(benefit, maximize=True)[2] - res.total <= res.bound + 1e-9              # within its guaranteed distance

score = TravelScore(agent_pos=rng.uniform(0, 10, (3, 2)), task_pos=rng.uniform(0, 10, (5, 2)), value=rng.uniform(1, 5, 5), speed=2.0, discount=0.9)
result = cbba(score, n_agents=3, n_tasks=5, adjacency=adjacency("ring", 3))               # each agent talks to its ring neighbours only
best, _ = brute_force_allocation(score, 3, 5)
assert result.converged and result.conflict_free and result.score >= 0.5 * best          # the published guarantee
```

### Coverage control

`CoverageDomain` samples a region on a grid (with an optional density and mask); `lloyd` moves every site towards the
density-weighted centroid of its Voronoi cell. The locational cost never increases, and the fixed points are centroidal
Voronoi configurations. For a uniform segment the optimum is known in closed form (`p_i = (2i − 1)/(2n)`, cost `1/(12 n²)`),
which is what the tests compare against.

<!-- docs-test -->
```python
import numpy as np
from aevion.multiagent import CoverageDomain, coverage_cost, lloyd

domain = CoverageDomain([0.0, 1.0], resolution=1 / 2000)
res = lloyd(np.array([[0.05], [0.1], [0.2]]), domain, iterations=2000)
assert np.allclose(np.sort(res.sites[:, 0]), [1 / 6, 1 / 2, 5 / 6], atol=2e-3)
assert abs(coverage_cost(res.sites, domain) - 1 / (12 * 9)) < 1e-4 and np.all(np.diff(res.cost) <= 1e-12)
```

### UTM-style deconfliction

* `detect_conflicts` finds, exactly, the time intervals in which two 4-D (piecewise-linear) trajectories are closer than a
  separation distance: relative motion is linear between breakpoints, so the squared distance is a quadratic whose roots are
  closed-form. Brief encounters between sample points cannot be missed.
* `schedule_departures` is strategic deconfliction by ground delay: routes are accepted in priority order, each with the
  smallest delay that leaves it conflict-free against those already accepted.
* `SpaceTimePlanner` plans every vehicle in priority order with space-time A\* (vehicles may wait) against the reserved paths of
  the others, with the same exact test, so a returned plan is conflict-free by construction. It reports why when an agent cannot
  be fitted (a sealed lane, a blocked goal, or a start inside another vehicle's separation).
* `predict_conflicts` is the tactical layer: closest-point-of-approach prediction from positions and velocities. Resolving a
  predicted conflict is the job of the ORCA and barrier filters in [Planning](#planning-and-avoidance).

<!-- docs-test -->
```python
import numpy as np
from aevion.multiagent import SpaceTimePlanner, Trajectory4D, detect_conflicts, schedule_departures
from aevion.planning import PlanningSpace

a = Trajectory4D([0.0, 10.0], [[0, 0, 0], [100, 0, 0]])
b = Trajectory4D([0.0, 10.0], [[100, 0, 0], [0, 0, 0]])
(conflict,) = detect_conflicts([a, b], 10.0)                                               # head-on: closer than 10 m from t = 4.5 to 5.5
assert abs(conflict.t_min - 5.0) < 1e-9 and abs(conflict.t_start - 4.5) < 1e-9
delays, ok = schedule_departures([a, b], separation=10.0, max_delay=60.0)
assert ok and delays[0] == 0.0 and delays[1] > 0.0                                         # the second flight is held on the ground

space = PlanningSpace(np.zeros((12, 5, 3), bool), (0.0, 0.0, 0.0), 1.0)
plan = SpaceTimePlanner(space, dt=1.0, separation=1.5).plan([[0.5, 2.5, 1.5], [10.5, 2.5, 1.5]], [[10.5, 2.5, 1.5], [0.5, 2.5, 1.5]])
assert plan.success and plan.conflicts(1.5) == []                                          # two vehicles swap places without a conflict
```

### Swarm laws in the engine

The `swarm` block of a configuration already gave each vehicle collision-avoidance offsets computed from the *last received*
position of its neighbours. `laws` adds cooperative behaviours on top, and every law sees only what the link delivers:
vehicle `i` knows itself exactly and the others as of their last delivered message, which may be old (`delay_s`), missing
(`loss`, `range_m`) or frozen (a `comm_blackout` fault). So the laws degrade the way a real decentralised swarm does.

| Law | Behaviour |
| --- | --- |
| `formation_keeping` | each vehicle shifts its reference towards the believed tracking errors of its neighbours: a vehicle that lags its slot is waited for, and the shape holds under a gust, a degraded rotor or a payload |
| `rendezvous` | consensus on position: vehicles gather at a common point |
| `coverage` | Lloyd's algorithm on the believed positions: each vehicle flies to the centroid of its Voronoi cell of a rectangle, optionally weighted by Gaussian hot spots |

```yaml
vehicle: {count: 6, formation: {type: ring, params: {spacing: 3.0}}}
swarm:
  avoid_radius: 1.0
  comm: {loss: 0.1, delay_s: 0.05}
  laws:
    - {type: formation_keeping, params: {gain: 0.4}}
faults:
  - {type: thrust_loss, t_start: 4.0, t_end: 9.0, vehicle: 2, params: {factor: 0.55}}
```

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation
from aevion.multiagent import formation_error, make_formation

offsets = make_formation("ring", 6, 3.0)

def shape_error(laws):
    cfg = {
        "duration": 12.0, "seed": 1,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 8.0], "radius": 6.0, "omega": 0.3}},
        "vehicle": {"count": 6, "formation": {"type": "ring", "params": {"spacing": 3.0}}},
        "swarm": {"avoid_radius": 1.0, "comm": {"delay_s": 0.05}, "laws": laws},
        "faults": [{"type": "thrust_loss", "t_start": 4.0, "t_end": 9.0, "vehicle": 2, "params": {"factor": 0.55}}],
    }
    run = Simulation(cfg).run()
    err = np.array([formation_error(p, offsets) for p in np.asarray(run["pos"])])
    return err[(run.t > 4.5) & (run.t < 9.0)].mean()

plain, kept = shape_error([]), shape_error([{"type": "formation_keeping", "params": {"gain": 0.4}}])
assert kept < 0.6 * plain                    # neighbours wait for the vehicle that lost thrust
```

> Note: What the guarantees are
>
> Consensus, Lloyd and CBBA are proven for their idealised models (a connected graph, reliable or symmetric links). The
> engine's link is lossy and delayed, so the laws are *tested* in simulation (the tests measure the formation error with and
> without the link and show that a dead link removes the law's benefit), not proven. `formation_keeping` is a
> reference-shifting heuristic, not a formation controller with a stability proof.

### Cooperative learning

`VecSwarmEnv(n_envs, n_agents=…, task=…)` simulates `n_envs` independent teams of quadrotors in one batch, with team episodes (a
crash or collision ends it for the whole team), neighbour-list observations (the `k` nearest agents inside `comm_range`, as a
decentralised policy would receive them) and three tasks: `formation`, `rendezvous` and `coverage`. `ParallelSwarmEnv` is the
dictionary-keyed API of PettingZoo's `ParallelEnv` (`reset() -> (obs, infos)`, `step(actions) -> obs, rewards,
terminations, truncations, infos`, `agents` emptied when the episode ends); it subclasses the real class if `pettingzoo` is
installed. `make_swarm_sb3_vec_env` exposes every agent as one environment of a Stable-Baselines3 `VecEnv` for
parameter-sharing training.

<!-- docs-test -->
```python
import numpy as np
from aevion.envs import VecSwarmEnv, parallel_env
from aevion.envs.swarm_policies import SwarmScriptedPolicy

env = VecSwarmEnv(4, task="formation", n_agents=4, k_neighbors=3, seed=0)
policy = SwarmScriptedPolicy(env)                      # a decentralised expert: it sees only one agent's observation
obs, successes = env.reset(seed=1), []
for _ in range(2 * env.max_steps):
    obs, reward, terminated, truncated, info = env.step(policy(obs))
    successes += info["success"][info["done"]].tolist()
assert np.mean(successes) == 1.0 and not info["crashed"].any()

pz = parallel_env(task="rendezvous", n_agents=3, seed=0)
obs, infos = pz.reset(seed=0)
obs, rewards, terminations, truncations, infos = pz.step({a: np.zeros(4) for a in pz.agents})
assert set(obs) == {"agent_0", "agent_1", "agent_2"} and all(isinstance(r, float) for r in rewards.values())
```

> Warning: Limits
>
> The cooperative bookkeeping (neighbour lists, the coverage cost) is NumPy; the dynamics run on any backend, but the
> swarm environment is not tensor-resident like `VecQuadEnv`. With few visible neighbours or a short `comm_range`, gathering
> can split the team into clusters: the tests show a sparse network failing where a fully connected one succeeds. PettingZoo
> itself is not installed in the development environment, so conformance was checked against the documented API by hand,
> not with `pettingzoo.test.parallel_api_test`.

---

<!-- page: bridges -->

## Bridges: MAVLink, PX4 logs, HIL and ROS 2

`aevion.bridges` connects the simulation to the tools around it. Everything here is implemented from public definitions and
is exercised in the test-suite against Aevion itself; none of it has been run against a real PX4, ArduPilot or ROS 2
installation (none was available). Read the caveats at the end of each section before trusting a link to
real software or hardware.

| Module | What it does |
| --- | --- |
| `aevion.bridges.mavlink` | MAVLink v1/v2 codec: framing, CRC, a streaming `Parser`, a table of message definitions |
| `aevion.bridges.telemetry` | `.tlog` reading and writing; `TelemetryLog` and `archive_to_tlog`; NED/FRD ↔ ENU/FLU conversions |
| `aevion.bridges.ulog` | PX4 ULog reader (`ULog`) and writer (`write_ulog`) |
| `aevion.bridges.hil` | `HilHook`: fly the simulated vehicle with an autopilot's motor outputs; `ReferenceAutopilot` for tests |
| `aevion.bridges.transport` | byte transports: in-process `LoopbackTransport`, loopback-only `UdpTransport` |
| `aevion.bridges.ros2` | ROS 2 message dictionaries and an optional `rclpy` publisher |

### MAVLink

The codec packs and unpacks messages from definitions in `MESSAGES` (field names, types, array lengths, extension fields). The
wire order (fields sorted by size, extensions last), the trimming of trailing zeros in MAVLink 2, and the CRC_EXTRA byte
are derived from those definitions and checked against the published constants (for example `HEARTBEAT` = 50, `ATTITUDE` = 39).
A frame whose definition differs from ours fails the checksum rather than decoding to nonsense. `Parser.feed(bytes)` accepts any
chunking of the byte stream, resynchronises after garbage, and counts `good`, `bad_crc` and unknown messages. Signed frames are
skipped over, not verified. Only the messages Aevion needs are defined (heartbeat, attitude, local and global position, HIL
sensor/GPS/actuator messages, and a few more); `define(name, id, (type, field), ...)` registers another one.

<!-- docs-test -->
```python
from aevion.bridges.mavlink import Parser, pack

frame = pack("ATTITUDE", time_boot_ms=1500, roll=0.1, pitch=-0.2, yaw=1.0, rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
(msg,) = Parser().feed(frame)
assert msg.name == "ATTITUDE" and abs(msg.fields["pitch"] + 0.2) < 1e-6
split = Parser()
assert split.feed(frame[:7]) == [] and len(split.feed(frame[7:])) == 1          # chunked input is fine
corrupt = bytearray(frame); corrupt[-3] ^= 0xFF
bad = Parser()
assert bad.feed(bytes(corrupt)) == [] and bad.bad_crc == 1                      # a damaged frame is counted, not decoded
```

### Telemetry logs and PX4 ULog

`archive_to_tlog(archive, path)` writes a simulated flight as a standard `.tlog` (big-endian microsecond timestamps before each
frame, as QGroundControl and MAVProxy write), converting the state to NED/FRD and the position to latitude and longitude about
`origin`. `TelemetryLog.from_tlog(path).to_archive()` goes the other way, so a flown log can be compared with a simulated one with
the ordinary analysis tools. Positions survive to about a millimetre and attitude to well under a degree, limited by the float32
fields and the millisecond time stamps.

`ULog(path)` reads PX4's logging format: formats (including nested messages, arrays, padding and strings), multi-instance topics, data,
parameters, info and dropouts; a truncated file returns what was written and sets `truncated`. `ULog.to_archive()` converts a PX4
flight (`vehicle_local_position`, `vehicle_attitude`, …) into an Aevion `RunArchive` in ENU/FLU, with the PX4 NED/FRD quaternion
converted by `ned_frd_to_enu_flu`. `write_ulog` produces a valid file for tests and for tools that need one.

<!-- docs-test -->
```python
import tempfile
from pathlib import Path

import numpy as np
from aevion.bridges.telemetry import TelemetryLog, archive_to_tlog
from aevion.engine import Simulation

run = Simulation({
    "duration": 4.0,
    "trajectory": {"type": "circle", "params": {"center": [0, 0, 6.0], "radius": 4.0, "omega": 0.5}},
    "logging": {"record_every": 4},
}).run()
with tempfile.TemporaryDirectory() as d:
    path = archive_to_tlog(run, Path(d) / "flight.tlog", rate_hz=20.0, origin=(47.3769, 8.5417, 408.0))
    log = TelemetryLog.from_tlog(path)
    replay = log.to_archive("replayed")
t = np.asarray(replay["t"])
z = np.interp(t, run.t - run.t[0], np.asarray(run["pos"])[:, 0, 2])
assert log.heartbeats > 3 and np.abs(np.asarray(replay["pos"])[:, 0, 2] - z).max() < 1e-3
```

### Hardware- and software-in-the-loop

`HilHook` follows the standard PX4/ArduPilot HIL interface. Every physics step it converts the sensors to `HIL_SENSOR` (IMU,
magnetometer, barometer) and `HIL_GPS`, sends them over a `Transport`, reads `HIL_ACTUATOR_CONTROLS`, and drives the rotors with
the returned outputs in place of the simulated controller. Options: `mapping` (`"thrust"`: ω = ω_max·√output, or `"speed"`:
ω = ω_max·output), `motor_map` (motor order), `require_armed` (zero outputs until the armed flag is set), `wait_s` (a lock-step
wait for a real autopilot's reply) and `send_every`. A NaN or missing output never reaches the dynamics.

`ReferenceAutopilot` is an autopilot built from Aevion's own ESKF and geometric controller that listens on the other end of a
`LoopbackTransport`; it is what the closed-loop tests use. With it the vehicle holds a point within about a metre, which is the
GNSS error, so the loop is correct; it says nothing about how a real firmware tunes its own gains.

<!-- docs-test -->
```python
import numpy as np
from aevion.bridges.hil import HilHook, ReferenceAutopilot
from aevion.bridges.transport import LoopbackTransport
from aevion.engine import Simulation
from aevion.vehicles.spec import VehicleSpec

ORIGIN = (47.3769, 8.5417, 408.0)
cfg = {
    "duration": 8.0,
    "trajectory": {"type": "hover", "params": {"pos": [0, 0, 5.0]}},
    "environment": {"origin": {"lat": ORIGIN[0], "lon": ORIGIN[1], "alt": ORIGIN[2]}},
    "sensors": ["imu", "mag", "baro", {"type": "gps", "params": {"rate": 10.0, "latency": 0.0}}],
}
sim_end, autopilot_end = LoopbackTransport.pair()
hook = HilHook(sim_end, origin=ORIGIN)
autopilot = ReferenceAutopilot(VehicleSpec.generic_1kg(), autopilot_end, origin=ORIGIN, target=(0.0, 0.0, 5.0))
run = Simulation(cfg, hooks=[hook]).run()
late = run.t > 5.0
assert hook.armed and hook.parser.bad_crc == 0 and autopilot.parser.bad_crc == 0
assert np.abs(run["pos"][late, 0] - [0, 0, 5.0]).max() < 1.0
```

Talking to a real autopilot uses `UdpTransport(local_port, remote_port)`. It binds and sends to the loopback interface only:
other hosts raise an error unless you pass `allow_remote=True`, so a simulation never opens a network port by accident.

> Warning: What is untested
>
> The message layouts follow the MAVLink common definitions, but the motor-output mapping and rates that PX4's or ArduPilot's HIL
> mode expect differ by firmware version and airframe. Check `mapping` and `motor_map` against your airframe first, with
> the propellers off. The reference loop uses an ideal sensor timeline; a real autopilot may need the lock-step `wait_s`.

### ROS 2

Aevion's frames already are ROS's REP 103 frames (ENU world, FLU body), so no axes are flipped; only the quaternion order changes
(ROS `x, y, z, w`) and the odometry twist is expressed in the body frame. `odometry_msg`, `imu_msg`, `pointcloud2_msg`,
`clock_msg` and `tf_msg` build plain dictionaries with the fields of `nav_msgs/Odometry`, `sensor_msgs/Imu`,
`sensor_msgs/PointCloud2`, `rosgraph_msgs/Clock` and `tf2_msgs/TFMessage`, so they can be inspected or serialised without ROS.

`Ros2Bridge` creates a node and publishers when `rclpy` is importable (and raises an `ImportError` that says what to do when
it is not), and `Ros2Hook(bridge, every=5)` publishes the simulated vehicle from inside a run. It is tested against a stand-in
for `rclpy`, not a ROS 2 installation, so the first real run is a check of QoS settings, topic names and the time source.

### System identification

See [System identification](#system-identification-1).

---

<!-- page: identification -->

## System identification

`aevion.identification.identify(spec, log, free=...)` fits a vehicle's physical parameters to a flight log, so a simulated vehicle
can be made to behave like a real one. It finds the values of `kT`, drag, inertia and motor time constant for which the
simulation reproduces the flight, and reports their uncertainty.

### Method

The method is multiple shooting. The log is cut into short windows (`window_s`, default one second, every `stride_s`). Each
window is simulated forward from the *recorded* state at its start with the *recorded* inputs, and the difference between the
simulated and recorded position, velocity, attitude and body rates along the window is the residual. Short windows keep the
problem well-conditioned (a long open-loop prediction of an unstable vehicle is hopeless), and all windows are simulated together
as one batch, which is what the batched dynamics are for. The multipliers are found by bounded nonlinear least squares in
log-space (so they stay positive), and the Gauss–Newton covariance `σ²(JᵀJ)⁻¹` gives standard errors and a correlation matrix.

The log is a dictionary (or archive) with `t`, `pos`, `vel`, `quat`, `omega` and either `cmd` (rotor-speed commands) or `rotor`
(measured rotor speeds). `load_csv_log` reads a CSV into that form. `noise` gives the measurement standard deviations used to
weight the residuals.

### What can and cannot be identified

| Parameter | Names | Condition |
| --- | --- | --- |
| thrust coefficient | `kT` | with the mass supplied (weigh the vehicle) |
| roll and pitch inertia | `Jxy` | given `kT` |
| linear and quadratic drag | `drag_lin`, `drag_quad` | needs flight at speed |
| motor time constant | `tau` | needs commands recorded at the physics rate (`record_every: 1`), not measured rotor speeds |
| yaw torque coefficient, yaw inertia | `kQ`, `Jz` | only the ratio `kQ / Jz` is observable: fit one with the other fixed |
| mass | `mass` | only `kT / mass` is observable: fit one with the other fixed |

`identify` does not hide this. The result's `correlation` matrix shows a pair that cannot be separated (an off-diagonal entry
near ±1), and `report()` prints a warning naming it. When the data fix only a ratio, the ratio is still recovered correctly.

<!-- docs-test -->
```python
import numpy as np
from aevion.engine import Simulation
from aevion.identification import identify
from aevion.vehicles.spec import VehicleSpec

nominal = VehicleSpec.generic_1kg()
rng = np.random.default_rng(0)
points = [[0, 0, 5]] + [[rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(3, 8)] for _ in range(6)]
flight = Simulation({
    "duration": 14.0,
    "seed": 3,
    "trajectory": {"type": "waypoints", "params": {"points": points, "speed": 3.0, "yaw_follow": True}},
    "logging": {"record_every": 1},                                    # commands at the physics rate
    "vehicle": {"model": "generic_1kg", "overrides": {"kT": nominal.kT * 0.88}},     # the "real" vehicle: 12 % less thrust
}).run()
log = {k: np.asarray(flight[k]) for k in ("t", "pos", "vel", "quat", "omega", "rotor", "cmd")}

fit = identify(nominal, log, free=("kT",))
assert fit.success and abs(fit.factors[0] - 0.88) < 0.01              # recovers the 12 % to within 1 %
assert fit.rms < 0.2 * fit.initial_rms                                # and the fitted model explains the log far better than the nominal one
```

### Results and how far to trust them

`IdentificationResult` has `names`, `factors` (multipliers of the nominal), `values` (the parameters themselves), `stderr`,
`correlation`, `rms` and `initial_rms`, `n_windows`, `n_residuals`, and `report()` for a text summary.

> Warning: Standard errors are optimistic
>
> The covariance assumes independent residuals with the given `noise`. Residuals along a window are strongly correlated, so
> the reported errors are smaller than the true scatter. In the test-suite, with five parameters free and a known truth, the
> estimates land within 1–4 % of the truth while the reported `stderr` is a fraction of that. Treat `stderr` as a lower bound and
> judge a fit by whether it explains a log it was *not* fitted to.

### Limits

* Identification fits the *model structure* of the simulator (rigid body, quadratic rotor thrust, first-order motor lag, linear
  and quadratic drag). A real vehicle with flexible arms, ground effect or a propeller that stalls will not be explained
  fully, and the parameters will absorb the difference.
* A log that does not excite an axis cannot identify it; hovering identifies almost nothing. Fly a zig-zag with yaw.
* With `use_measured_rotors=True` (or a log without `cmd`) the motor lag is excluded from the fit and `tau` cannot be fitted.

---

<!-- page: ml -->

## Machine learning guide

### Environments

| Class | Use |
|---|---|
| `QuadrotorEnv` (Gymnasium) | single drone, NumPy `float32`; ids `Aevion-{Hover,Waypoints,Tracking,Navigation}-v0` |
| `VecQuadEnv` | batched, tensor-native: returns arrays on the chosen backend (NumPy, torch CPU, CUDA) |
| `make_sb3_vec_env` | `VecQuadEnv` wrapped as a Stable-Baselines3 `VecEnv` |
| `QuadVectorEnv` | `VecQuadEnv` as a Gymnasium `VectorEnv` (same-step auto-reset, `info["final_obs"]`) |
| `VecSwarmEnv`, `ParallelSwarmEnv` | cooperative teams of quadrotors, and the PettingZoo-style API (see [Multi-agent](#cooperative-learning)) |

```python
from aevion.envs import VecQuadEnv
env = VecQuadEnv(4096, task="navigation", backend="cuda", action_mode="velocity", lidar=True)
obs = env.reset(seed=0)                       # torch.Tensor on cuda
obs, reward, terminated, truncated, info = env.step(policy(obs))
```

Finished environments reset automatically; `info["final_observation"]` holds the terminal observation (needed for
correct value bootstrapping on truncation) and `info["final_return"]` the episode return.

### Actions

| `action_mode` | Action (all in [−1, 1]) | Low-level loop |
|---|---|---|
| `rotor` | one value per rotor; 0 = hover speed | none (raw actuators) |
| `thrust_rates` | `[collective thrust, p*, q*, r*]` | body-rate controller (typical for racing/agile RL) |
| `velocity` | `[vx, vy, vz, yaw-rate]` × `max_speed` | cascaded PID velocity loop |
| `position` | relative `[dx, dy, dz, dψ]` | geometric controller |

### Observations

`[target − p, v − v_target, vec(R), ω, previous action, (LiDAR ranges)]`, scaled by `obs_scales`. Optional Gaussian
observation noise comes from `DomainRandomization.obs_noise`.

### Tasks

`hover`, `waypoints` (sequence of goals), `tracking` (circle / figure-8 with random size, rate, phase),
`navigation` (reach a goal through obstacles using LiDAR), `connected_navigation` (reach a goal while staying connected to a cellular network: six extra link features in the observation and a link-quality/outage term in the reward, see [Wireless](#communication-aware-learning)). The additional tasks are `gate_racing`, `landing`, `target_tracking`, `inspection` and `area_coverage` (below). Write your own by subclassing `envs.tasks.Task`.

### Domain randomisation

```python
from aevion.envs import DomainRandomization
dr = DomainRandomization(mass=0.2, kT=0.1, tau=0.3, drag=0.5, wind_speed=6.0, obs_noise=0.02, rotor_failure_prob=0.05)
env = VecQuadEnv(1024, domain_randomization=dr)
```

Parameters are re-drawn per environment at each of that environment's resets. Controllers keep using the
*nominal* model, so randomised plants create real model mismatch.

### Differentiable simulation

The torch backend is autograd-friendly (smooth dynamics, ground contact via `where`):

```python
sim = Multirotor("generic_1kg", n=1, backend="torch", dtype="float64", kernel="reference")
scale = torch.tensor(1.02, requires_grad=True)
s = sim.reset(pos=[0, 0, 2.0])
for _ in range(100):
    s = sim.step(s, sim.hover_speed() * scale * torch.ones(1, 4, dtype=torch.float64))
s.pos[0, 2].backward()                      # d z_final / d scale, verified against finite differences
```

Use it for gradient-based policy learning, trajectory optimisation and parameter identification. Gradients are not
defined through discrete events (collisions switching on/off).

### Evaluation

```python
from aevion.experiments import evaluate_policy
res = evaluate_policy(lambda obs: model.predict(obs)[0], lambda seed: VecQuadEnv(32, seed=seed),
                      seeds=range(10), episodes_per_seed=32)
print(res.text())            # mean [95 % CI] over seeds for return, success rate, crash rate, ...
```

Statistics are taken over independent seeds (not over episodes of one seed), which is what a paper should report.
`aevion.analysis.figures_stats.fig_learning_curves` plots mean ± CI learning curves from a `TrainingLog`.

### Additional tasks

| Task | Goal | Reward | Success |
| --- | --- | --- | --- |
| `gate_racing` | pass through circular gates in order (a gate counts when the path crosses its plane forwards within its radius) | progress to the next gate, a bonus per gate, a penalty for crossing outside the radius | all gates passed |
| `landing` | touch down on a platform that drifts across the ground | funnel shaping (be low only when aligned), relative-velocity penalty, bonus on success | touchdown inside the zone and slow; anywhere else (or too fast) is a failure |
| `target_tracking` | stay with a target whose velocity is a correlated random process (unlike the scripted `tracking` curves) | `exp(−2 error)` plus a velocity match term | within `success_radius` of the target |
| `inspection` | come within `sense` of points on a vertical cylinder; flying into it crashes | +3 per newly inspected point | all points inspected |
| `area_coverage` | visit every cell of a square area | newly covered cells, altitude hold | `done_fraction` of the cells covered |

Every task has `set_difficulty(level)` with `level ∈ [0, 1]` (narrower gates, faster platforms, a faster target, a smaller sensing
radius or footprint), and tasks with their own failure modes implement `extra_crash(env)`, which the environment ORs into its
crash flag. Observation features are appended after the standard ones (`extra_obs_dim`).

Each task ships with a privileged scripted expert (`aevion.envs.experts.make_expert(env)`, velocity action mode) that reads the simulator state
(gate normals, the platform's velocity, which points remain): the teacher in a teacher-student set-up, and the reference the
tests hold the tasks to.

### Curriculum

`Curriculum(stages, promote_at, window)` raises the difficulty when the success rate over the last `window` finished episodes
reaches `promote_at` (and optionally lowers it below `demote_at`). A stage is a level for `set_difficulty` or a dictionary of
task attributes. It never advances faster than one stage per window, so a lucky streak cannot skip levels.

### Imitation learning, datasets and export

`collect(env, policy, steps, labeler=…, beta=…)` rolls a policy out and stores `(obs, action, reward, terminal, timeout, next_obs)` rows
in a `Dataset` with the layout of D4RL (`to_d4rl()` / `from_d4rl()`, `.npz` files). Terminals (a crash, a success) and timeouts
(the time limit) are kept apart, as offline RL needs.

`behavior_cloning` fits a `tanh(MLP)` policy to expert actions. Trained only on states the expert visits, it makes small errors
that carry it to states the expert never showed, and the errors compound (covariate shift). DAgger (Ross, Gordon and Bagnell
2011) repairs this: iteration 0 collects expert demonstrations; later iterations run the *learner* (mixed with the expert with
probability `β₀ᶦ`), let the expert label every state visited, aggregate and retrain.

`MLPPolicy` normalises its input inside the module, so `export_onnx` and `export_torchscript` produce graphs that take raw
observations and return actions in `[-1, 1]`; `verify_onnx` runs the exported graph with ONNX Runtime and checks it against the
PyTorch policy.

<!-- docs-test -->
```python
import tempfile
from pathlib import Path

import numpy as np
from aevion.envs import VecQuadEnv
from aevion.envs.experts import make_expert
from aevion.ml import collect, dagger, evaluate_policy_rollouts, export_onnx
from aevion.ml.policies import verify_onnx


def landing_env(seed):
    return VecQuadEnv(16, task="landing", action_mode="velocity", seed=seed, max_episode_s=14.0, control_dt=0.05, physics_dt=0.01)

env = landing_env(0)
expert = make_expert(env)                              # the privileged teacher
demo = collect(env, expert, 40, seed=1)
assert len(demo) == 16 * 40 and demo.to_d4rl()["observations"].shape[1] == env.obs_dim

result = dagger(env, expert, iterations=1, steps_per_iter=250, hidden=(64, 64), epochs=40, eval_env=landing_env(5), eval_episodes=32)
assert result.history[0]["dataset"] == 16 * 250 and result.history[1]["dataset"] == 2 * 16 * 250
assert result.history[1]["success_rate"] >= 0.85       # a few hundred states of aggregated data are enough to land

with tempfile.TemporaryDirectory() as d:
    path = export_onnx(result.policy, Path(d) / "landing.onnx")
    assert verify_onnx(result.policy, path) < 1e-5     # the exported graph reproduces the PyTorch policy
```

> Note: Two pitfalls this code avoids
>
> A feature that never varies in the demonstrations (an unused action channel fed back as an input) must not be amplified when
> the learner first moves it: the observation normaliser floors every standard deviation at 0.05. Without the floor, cloning a
> scripted expert produced a policy that saturated within ten steps. And the expert's decision rule should be *continuous*: the
> landing expert descends along a funnel instead of switching between "hold" and "descend", which a regressor cannot imitate.

### Tracking, tuning and parallel rollouts

* Tracking: `JsonlLogger` and `CsvLogger` need nothing; `TensorBoardLogger`, `MLflowLogger` and `WandbLogger` import their library
  when constructed and say what to install otherwise. `MultiLogger` fans one call out to several, and `LoggingCallback` adapts a
  logger to Stable-Baselines3 training. The three optional adapters are tested against stand-ins for the libraries (none is
  installed in the development environment), not against the real packages.
* Tuning: `optuna_search(objective, space, n_trials)` takes a plain search space (`{"wn": ("float", 1, 15)}`) and any objective
  (a controller gain, a reward weight, a learning rate); trials that raise are skipped and the study is seeded.
* Parallel rollouts: `parallel_collect(make_env, make_policy, steps, n_workers)` collects in worker processes and merges the
  datasets deterministically. Ray and Dask runners are not provided; the two factories are the units of work such a runner would
  distribute.

### Vision datasets

`aevion.datasets.vision.VisionRecorder` writes PNG images, 16-bit depth, segmentation, COCO and YOLO labels and optical-flow ground
truth from the simulated camera during a run: see [Vision datasets](#vision-datasets-1).

### Throughput tips

* CPU: use the NumPy backend with Numba (`pip install aevion[fast]`); batch ≥ 1024 environments.
* GPU: `backend="cuda"`, large batches, float32. Keep policies on the same device.
* `control_dt` / `physics_dt` set the decimation; a coarser `physics_dt` (0.01 s) roughly doubles speed at some accuracy cost.

---

<!-- page: vision -->

## Vision datasets

A simulated camera knows the exact geometry, so its labels are exact rather than annotated. `aevion.datasets.vision` turns the
ray-cast camera of [Sensors](#radar-lidar-and-camera) into a labelled image dataset:

| Output | How it is obtained |
| --- | --- |
| `images/NNNNNN.png` | the RGB image (8-bit PNG, written without an image library) |
| `depth16/NNNNNN.png` | z-depth in millimetres as 16-bit PNG (0 = no return) |
| `seg/NNNNNN.png` | object id + 1 per pixel (0 = background), primitives of one mesh share an id |
| `annotations.json` | COCO detection format: `images`, `annotations` (`bbox` as `[x, y, w, h]`, `area` = visible pixels), `categories` |
| `labels/NNNNNN.txt`, `data.yaml` | YOLO text labels (`class cx cy w h`, normalised) and the dataset file |
| `flow/NNNNNN.flo` | optical flow from this frame to the next, Middlebury `.flo` (NaN where undefined) |

Boxes come from the per-pixel primitive ids, so a box covers exactly the *visible* part of an object (an occluded object gets a
box around what shows). Optical flow is computed from the depth image and the two camera poses: every pixel is back-projected into
the world, seen from the next pose and projected again, which is exact for a rigid static scene. With a second depth image the
function also reports which pixels are occluded in the next frame.

<!-- docs-test -->
```python
import json
import tempfile
from pathlib import Path

import numpy as np
from aevion.datasets.vision import VisionRecorder, read_flo, read_png
from aevion.engine import Simulation

cfg = {
    "duration": 2.0,
    "world": {"type": "scene", "params": {"boxes": [{"center": [14, 0, 2], "size": [2, 3, 4]}], "spheres": [{"center": [10, 2.5, 2.0], "radius": 1.0}], "bounds": [-5, -15, 0, 40, 15, 20]}},
    "trajectory": {"type": "polyline", "params": {"points": [[0, 0, 2], [4, 0, 2]], "speed": 2.0}},
    "sensors": [{"type": "camera", "params": {"rate": 10.0, "width": 64, "height": 48, "modes": ["depth", "rgb", "seg"], "far": 40.0}}],
}
with tempfile.TemporaryDirectory() as d:
    recorder = VisionRecorder(Path(d) / "dataset", camera="camera", every=1, max_frames=50)
    Simulation(cfg, hooks=[recorder]).run()
    ds = Path(d) / "dataset"
    coco = json.loads((ds / "annotations.json").read_text())
    assert len(coco["images"]) == recorder.frames and {c["name"] for c in coco["categories"]} == {"box", "sphere"}
    assert all(0 <= a["bbox"][0] and a["bbox"][0] + a["bbox"][2] <= 64 for a in coco["annotations"])
    depth_mm = read_png(ds / "depth16" / "000000.png")            # uint16 millimetres
    assert depth_mm.dtype == np.uint16 and depth_mm.max() > 5000
    flow = read_flo(ds / "flow" / "000000.flo")                    # (H, W, 2) pixels, NaN where undefined
    assert flow.shape == (48, 64, 2) and np.isfinite(flow).any()
```

From the command line: `aevion vision-dataset config.yaml --out dataset --every 2 --max-frames 500`.

### How exact is the flow?

The tests check the optical flow against closed forms: a fronto-parallel plane approached along the optical axis expands about the
principal point by `Z / (Z − d)`; a sideways step shifts it by `f t / Z`; a pure rotation gives a field independent of depth that
equals the homography `K Rᵀ K⁻¹`. They also render two frames of a scene with the real camera and check that the flow lands
97 % or more of the non-occluded pixels on the same surface (identified by the segmentation id) in the next frame; the rest are
pixels on object edges.

> Warning: What the labels are not
>
> The images are the stylised ray-cast renderings of the camera model: no textures, simple lighting, first surface only.
> A detector or flow network trained on them will meet a domain gap on real images. Moving obstacles are not rigid with
> respect to the world, so the flow of their pixels is wrong by the obstacle's own motion. Transparent surfaces and mesh
> materials are not modelled, and the ground has no segmentation object (it is background).

---

<!-- page: experiments -->

## Experiments

The experiment toolkit covers sensitivity analysis (which parameters matter), Bayesian optimisation (which values are best), grids with
statistics and leaderboards (which method is better, and whether the difference is real), and caching, sharding and a result database
for running thousands of trials without losing work.

| Module | Provides |
| --- | --- |
| `design` | Latin hypercube (maximin), Sobol, Halton, full factorial; `ParameterSpace` |
| `sensitivity` | Morris elementary effects, Sobol first-order and total indices with bootstrap intervals |
| `bayesopt` | Gaussian-process regression and Bayesian optimisation (EI, LCB, PI; batches) |
| `config_experiment` | `ConfigExperiment`: methods × conditions × seeds over full `Simulation` configurations |
| `cache`, `runners`, `database` | content-addressed result cache, serial / process / Slurm runners, SQLite database |
| `suites` | benchmark suites and leaderboards with significance |

### Designs and parameter spaces

A `ParameterSpace` names the parameters (`(lo, hi)` uniform, `("log", lo, hi)` log-uniform, `("int", lo, hi)`, or a list of
categories) and maps unit-cube designs to them. Space-filling designs cover the box far more evenly than random draws: the
discrepancy of a 64-point Sobol design is under half that of random points, and a Latin hypercube has exactly one point in each of
the `n` strata of every coordinate (the tests assert both).

### Sensitivity analysis

`morris` screens many parameters cheaply (`r (d + 1)` runs): `μ*` ranks importance and `σ` flags nonlinearity or interactions.
`sobol_indices` decomposes the output variance (`n (d + 2)` runs): the first-order index `S_i` is the share explained by parameter `i`
alone, the total index `S_Ti` includes its interactions, and `S_Ti − S_i` is the part that only acts through interactions. The
design and the analysis are separate (`saltelli_design` / `sobol_analyze`), so the expensive evaluations can be run by anything.
The estimators are the Saltelli (2010) and Jansen (1999) ones; the tests recover the analytic indices of the Ishigami function
(S₁ = 0.314, 0.442, 0; S_T = 0.558, 0.442, 0.244) and of the Sobol g-function within 0.03.

<!-- docs-test -->
```python
import math
import numpy as np
from aevion.experiments import morris, sobol_indices

def ishigami(X, a=7.0, b=0.1):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])

box = [(-math.pi, math.pi)] * 3
res = sobol_indices(ishigami, box, n=4096, vectorized=True)
assert np.allclose(res.s1, [0.3139, 0.4424, 0.0], atol=0.04) and np.allclose(res.st, [0.5576, 0.4424, 0.2437], atol=0.04)
assert res.interactions()[2] > 0.15                  # x3 matters only through its interaction with x1

screen = morris(lambda x: 3 * x[0] + x[1], [(0, 1)] * 3, r=12, seed=0)
assert np.allclose(screen.mu_star, [3, 1, 0]) and list(screen.ranking()[:2]) == [0, 1]
```

### Bayesian optimisation

`GaussianProcess` is a standard GP on standardised outputs (Matérn-5/2 or squared-exponential, one length scale per input),
fitted by maximising the log marginal likelihood; the tests compare its likelihood and posterior with the textbook formulas.
`bayes_opt` starts from a Sobol design and proposes points by maximising Expected Improvement (`acquisition="lcb"` and `"pi"` are
available, `batch=q` proposes `q` points per iteration with the constant-liar heuristic). A diverged evaluation (a non-finite
value) is replaced by the worst value seen and counted in `failures`, and the optimiser never proposes the same point twice.

<!-- docs-test -->
```python
import math
from aevion.experiments import bayes_opt

def branin(x):
    b, c, t = 5.1 / (4 * math.pi**2), 5 / math.pi, 1 / (8 * math.pi)
    return (x[1] - b * x[0] ** 2 + c * x[0] - 6) ** 2 + 10 * (1 - t) * math.cos(x[0]) + 10

result = bayes_opt(branin, [(-5, 10), (0, 15)], n_init=8, n_iter=22, seed=0)      # 30 evaluations
assert result.y_best < 0.4 + 0.3 and len(result.y) == 30                           # global minimum 0.3979
```

### Experiments over configurations

`ConfigExperiment(name, base, methods, conditions, seeds)` builds one complete `Simulation` configuration per trial: the base,
deep-merged with the condition's overrides and then the method's (a component block whose `type` differs replaces the base's
block instead of merging with it), plus the seed. Anything a configuration can express is therefore a factor: controller,
estimator, swarm law, faults, wind, sensors. The metric is a *registered name* (`flight` by default), so the experiment is plain
data: `save_spec` / `load_spec` write it to JSON, and nothing in the file is executed.

* Cache: A trial's key is the BLAKE2b hash of its canonical configuration, metric and Aevion version, so any change gives a new key and a stale result
  can never be served. Entries are written atomically and a damaged entry is treated as a miss. Each trial is stored the moment it finishes, so an **interrupted run
  resumes where it stopped** (the tests kill a run and check the resumed result equals an uninterrupted one).
* Failures: A trial that raises (an unreachable goal, a diverged simulation) becomes a `failed` row with its error; it is not cached, so a re-run retries it.
* Runners: `SerialRunner`, `ProcessRunner(workers)`; `run(shard=(k, n))` executes trials with `index % n == k`, and
  `slurm_script` writes a job-array script whose tasks run `python -m aevion.experiments.worker spec.json --cache DIR --shard $SLURM_ARRAY_TASK_ID --of N`
  into a shared cache. Ray and Dask runners are not provided: anything with an `imap(fn, jobs)` method works as a runner.
* Database: `ResultDB` is SQLite from the standard library: rows keyed by trial hash (re-adding updates, merging deduplicates),
  numeric metrics in a queryable table, `best`, `metric_values`, `failures`, `merge`.
* Design experiments: `ConfigExperiment.from_design(...)` makes one condition per design point from a `ParameterSpace` and a map from parameter
  names to dotted configuration paths; `design_table` returns `(X, y)` ready for sensitivity analysis or a surrogate model.

<!-- docs-test -->
```python
import tempfile
from pathlib import Path

from aevion.experiments import ConfigExperiment, ResultCache, ResultDB, leaderboard

base = {"duration": 2.5, "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.7}}}
experiment = ConfigExperiment(
    "controllers", base,
    methods={"geometric": {"controller": "geometric"}, "cascaded": {"controller": "cascaded"}},
    conditions={"calm": {}, "windy": {"wind": {"type": "constant", "velocity": [4.0, 0.0, 0.0]}}},
    seeds=[0, 1], metric_params={"skip": 0.5},
)
with tempfile.TemporaryDirectory() as d:
    cache = ResultCache(Path(d) / "cache")
    first = experiment.run(cache=cache)
    again = experiment.run(cache=cache)
    assert first.meta["computed"] == 8 and again.meta["computed"] == 0 and again.meta["cached"] == 8   # a repeat costs nothing
    with ResultDB(Path(d) / "results.db") as db:
        db.add_result(first)
        assert db.best("controllers", "rmse_pos")["rmse_pos"] == first.values("rmse_pos").min()
    board = leaderboard(first.rows, "rmse_pos")                       # ranked, with each method's p-value against the leader
    assert board[0]["rank"] == 1 and board[0]["p_vs_best"] is None
```

### Benchmark suites and leaderboards

A `BenchmarkSuite` fixes scenarios, seeds and the ranking metric so methods from different studies are compared on the same footing;
`run_suite(suite, methods)` gives a leaderboard per scenario and an overall ranking by mean rank. `leaderboard` ranks by the mean and,
because a ranking without a significance statement over-reads noise, also reports each method's Holm-corrected Mann–Whitney p-value
against the leader and marks the methods that cannot be told apart from it. Two suites are built in (`tracking`, `disturbance`);
`aevion suite run tracking --controllers geometric cascaded` runs one from the command line.

> Note: Limits
>
> The Gaussian process is exact (cubic in the number of evaluations): it suits the hundreds of evaluations of a simulation study, not
> tens of thousands. Sobol indices need enough samples (a few thousand per factor for stable total indices) and assume independent
> inputs. Sharding assigns trials by index, which balances the *count* of trials, not their cost.

---

<!-- page: analysis -->

## Analysis and research-grade figures

The figure style (`aevion.analysis.style`) covers the column sizes and typefaces of four publishers. The module also provides the diagnostic figures that control,
estimation and classification papers need, and a generator for the methods appendix.

### Journal templates

| Venue (`venue=`) | Column widths | Typeface class |
| --- | --- | --- |
| `ieee` | 3.50 in (88.9 mm) single, 7.16 in double | serif (Times) |
| `acm` | 3.33 in single, 7.00 in double | serif (Linux Libertine if installed, else Times) |
| `elsevier` | 3.54 in (90 mm) single, 5.51 in (140 mm) one-and-a-half, 7.48 in (190 mm) double | sans-serif (Arial / Helvetica) |
| `springer` | 3.31 in (84 mm) single, 6.85 in (174 mm) double | serif (Times) |
| `plain` | 3.40 in single, 6.80 in double | serif |

Every figure function takes `venue` and `columns` (`"single"`, `"double"`, and `"one_half"` where the venue has it). The widths come from the
publishers' author guidelines as of writing: check them against the template you submit to. `venue_summary(venue)` prints the table
row, and `figure_latex(path, caption, label, venue, columns)` writes the `figure` (or `figure*`) environment with the printed width,
the venue's placement and, for ACM, the mandatory `\Description` (alternative text).

### Diagnostic figures

| Function | Shows |
| --- | --- |
| `fig_bode(system)` | magnitude and phase against frequency, with gain and phase margins marked; `system` is `(num, den)`, `(A, B, C, D)` or a SciPy LTI object, or a dict of them |
| `fig_pzmap(system, discrete=False)` | poles and zeros with the stability boundary (imaginary axis or unit circle); unstable poles are flagged |
| `fig_allan(taus, adev, fit)` | Allan deviation with the asymptotes of angle random walk, bias instability and rate random walk |
| `fig_nees(t, nees, dof)` | NEES against time with the χ² acceptance band and the fraction of samples inside it |
| `fig_roc({name: (y, score)})` | ROC curves with AUC (and optional bootstrap intervals) |
| `fig_ridgeline({name: samples})` | stacked kernel-density curves by group, medians marked |
| `fig_step_response(t, y, target)` | step response with the settling band and rise time, overshoot and settling time |

The numbers behind them live in `aevion.analysis.diagnostics` and are tested on their own: a first-order lag is −3.01 dB and
−45° at its corner; the loop `2 / (s (s+1) (s+2))` has gain margin exactly 3 at `ω = √2`; the ROC curve and AUC agree with
scikit-learn (ties included), and the AUC of two unit-variance Gaussians separated by `d′` is `Φ(d′/√2)`.

<!-- docs-test -->
```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
from aevion.analysis.diagnostics import auc, margins
from aevion.analysis.figures_diag import fig_bode, fig_roc
from aevion.analysis.templates import figure_latex, venue_summary

loop = ([2.0], [1.0, 3.0, 2.0, 0.0])                       # L(s) = 2 / (s (s + 1) (s + 2))
m = margins(loop)
assert abs(m["gain_margin"] - 3.0) < 1e-6 and abs(m["w_pc"] - 2 ** 0.5) < 1e-6
fig = fig_bode(loop, venue="elsevier")                      # Elsevier asks for a sans-serif face at 90 mm
assert abs(fig.get_size_inches()[0] - 3.54) < 1e-9

y = np.r_[np.ones(100), np.zeros(100)].astype(bool)
scores = np.r_[np.random.default_rng(0).normal(1.5, 1, 100), np.random.default_rng(1).normal(0, 1, 100)]
assert 0.8 < auc(y, scores) < 1.0
assert "\\Description{" in figure_latex("fig/bode", "Open-loop response.", "fig:bode", "acm")
assert venue_summary("ieee")["widths_in"]["single"] == 3.5
```

### Methods appendix

`methods_appendix(config=..., result=..., experiment=..., metrics=[...])` writes the *Simulation*, *Experimental design*, *Metrics*,
*Statistical analysis* and *Reproducibility* sections from the actual configuration and experiment record, not from a template with
blanks: the integrator and step, vehicle, controller, estimator, sensors, wind and faults, the number of methods, conditions, seeds
and runs, the definition of each reported metric, the tests used (Welch, Mann–Whitney, Holm, Cohen's d, Cliff's delta) and the Aevion
and library versions. `.markdown()` and `.latex()` (escaped, with `itemize` lists) give the two forms, and `aevion methods config.yaml`
prints it. The wording is plain and factual, a starting point to edit rather than finished prose.

<!-- docs-test -->
```python
from aevion.analysis.appendix import methods_appendix

appendix = methods_appendix(config={"duration": 8.0, "wind": {"type": "constant", "velocity": [3.0, 0.0, 0.0]}}, metrics=["rmse_pos", "energy_wh"])
text = appendix.markdown()
assert "`rk4`" in text and "8 s per run" in text and "constant" in text and "`rmse_pos`: root-mean-square 3-D position error" in text
assert "\\subsection*{Simulation}" in appendix.latex() and "rmse\\_pos" in appendix.latex()
```

---

<!-- page: viewer -->

## Viewer

The Aevion viewer turns a run archive into an interactive 3-D replay that opens in any modern browser. It runs on the *viewer's*
GPU (integrated graphics are fine; software rendering works too), so the Python side never needs a GPU, a window system or a 3-D
library. Nothing is downloaded: the web app and a pinned copy of three.js 0.160.0 ship inside the package, so every mode works
offline.

| You want | Use |
| --- | --- |
| Send someone a replay | `aevion replay run.aevd` → one self-contained `.html` file |
| Look at a long run (hours, many vehicles) | `aevion replay run.aevd --serve` |
| Watch a simulation while it runs | `aevion live scenario.yaml --open` |
| Compare controllers or parameters | `aevion replay a.aevd --compare b.aevd c.aevd --label PID --label MPC --label LQR` |
| A figure or a video for a paper/talk | `aevion replay run.aevd --screenshot fig.png --theme print`, `--video clip.mp4` |
| A notebook cell | `from aevion.viewer import show; show(archive)` |
| Build a scene by clicking | `aevion edit --open` → download `scene.json` → `aevion run scene.json` |

### What you see

* Scene: terrain (heightmap or checkered ground, optional imagery), buildings, trees, spheres, a procedural sky with a time of
  day, soft shadows. Vehicles are built from the vehicle specification (arm layout, spin directions, propeller size), with spinning
  propellers driven by the recorded rotor speeds; a glTF/GLB model can replace the procedural body.
* Layers (toggle in the drawer or with a key): trails coloured by speed, altitude, link SINR or battery; reference path and
  planned waypoints; velocity/wind/thrust vectors and rotor wash; LiDAR point cloud and an accumulated occupancy map; camera
  frustums; radio cells with sector wedges, links coloured by SINR, outage halos and handover / link-failure pulses; antenna
  pattern lobes; SINR coverage slices or a stacked volume; ghost vehicles of compared runs.
* Charts and timeline: any recorded or derived channel as a time-series, synchronised with playback; click or drag to scrub,
  wheel to zoom every chart at once; events and link outages are drawn on the timeline.
* Cameras: orbit, chase, top-down and first-person (keys `1`–`4`), follow mode, and a deep link (`#t=5.5&cam=chase&play=0`).
* Themes: dark, light and print (white page, serif type) for figures; `ui=0` hides every panel.

Keyboard: `Space` play/pause, `←/→` step (Shift: ten frames), `[` `]` speed, `R` restart, `1`–`4` camera, `F` follow, `Tab` next
vehicle, `T V L W C` layers, `H` hide the interface, `D` drawer, `S` screenshot, `?` help.

### Replays

<!-- docs-test -->
```python
from aevion.scenario import Scenario
from aevion.viewer import export_html, build_payload

archive = Scenario(trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.2, omega=1.0), duration=3.0, record_every=4).run()
path = export_html(archive, "replay.html")          # one file: the app, three.js and the data
payload = build_payload(archive, max_frames=100)    # the same data as a dict (aevion.view/2)
assert path.stat().st_size > 500_000 and payload["format"] == "aevion.view/2" and payload["meta"]["T"] <= 100
```

`export_html` decimates to `max_frames` (default 2400) frames, keeps the first and last, and embeds LiDAR scans, the world, radio
cells, coverage maps and antenna lobes when the run has them. Large optional channels are dropped (and listed in
`meta["dropped_channels"]`) once they exceed `extras_budget_mb`. For runs too long to embed, serve them instead.

### Long runs: the local server

<!-- docs-test -->
```python
import json
import http.client
from aevion.viewer import ViewerServer
from aevion.viewer.protocol import decode_bundle

with ViewerServer(archive) as server:               # binds to 127.0.0.1 on a free port
    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    conn.request("GET", "/api/meta")
    meta = json.loads(conn.getresponse().read())
    conn.request("GET", "/api/frames?names=pos,soc&start=0.5&stop=1.5&max=50")
    header, arrays = decode_bundle(conn.getresponse().read())
assert "pos" in meta["channels"] and arrays["pos"].shape[0] <= 50
```

The page first loads a coarse overview of the whole run (up to 4000 frames), then fetches a fine window around the playhead as
you scrub, so a one-hour run with many vehicles stays responsive and memory stays bounded. The server is read-only and local by
design: it binds to loopback, rejects requests whose `Host` header does not name it (DNS rebinding), rejects WebSocket upgrades from
other origins, serves only the packaged web files (path-contained, type allow-listed), bounds every request parameter, and sends a
restrictive Content-Security-Policy. Use `--host 0.0.0.0` only on a network you trust: it is plain HTTP.

### Live view

`aevion live scenario.yaml` runs the simulation and streams every recorded frame to the browser over a WebSocket. The run starts
when the first browser connects (`--no-wait` to start at once); `--speed 2` runs at twice real time, `--speed 0` as fast as the
machine allows; `--keep` keeps the server up afterwards so the finished run can still be inspected. A browser that connects late
or reloads is caught up from a replay buffer (256 MB; beyond that new frames are still delivered live but not retained). The
`LIVE` button follows the newest data; scrub back at any time to look at the past.

```python
from aevion.viewer import serve_live

archive = serve_live("examples/scenarios/wireless_city.yaml", speed=1.0, open_browser=True)
```

Handovers and radio-link failures appear as events as they happen. Streaming costs the simulation little: frames are stacked and
sent in chunks of up to 64 frames or 100 ms.

### Comparing runs

<!-- docs-test -->
```python
from aevion.viewer import compare_html

calm = archive
windy = Scenario(trajectory=dict(kind="circle", center=[0, 0, 2.5], radius=1.2, omega=1.0), duration=3.0, record_every=4,
                 wind={"type": "ConstantWind", "velocity": [3, 0, 0]}).run()
page = compare_html([calm, windy], "compare.html", labels=["calm", "windy"])
assert page.exists()
```

The first run is the main vehicle; the others are translucent ghosts placed by *time*, so runs of different length or sampling can
be compared (a run that ends early holds its last pose). The drawer lists, for each ghost, its maximum and RMS distance to the main
run and the first instant it deviates by more than one metre, with a difference chart you can scrub.

### Figures, screenshots and video

```bash
aevion replay run.aevd --screenshot fig.png --theme print --size 1600x900 --t 12.5
aevion replay run.aevd --video clip.mp4 --fps 30 --speed 2 --camera orbit --sweep 90
aevion replay run.aevd --video clip.gif --size 640x360
aevion replay run.aevd --video frames/            # a directory of frame_00001.png ...
```

The frames come from the real viewer in a headless Chrome, Edge or Chromium, each rendered at an explicit simulation time
(`__aevion.renderFrame(t)`), so a video is deterministic and does not depend on how fast the machine is. Requirements: a
Chromium-family browser (set `AEVION_BROWSER` to pick one), Pillow for `.gif`, `ffmpeg` on `PATH` (or `AEVION_FFMPEG`) for
`.mp4` / `.webm`. Videos can include the charts and panels (`--ui`). In Python: `render_video`, `render_frames`, `screenshot`.

### Notebooks

`show(archive)` embeds the viewer in a Jupyter or VS Code notebook cell. `mode="inline"` (default) embeds the page itself (works on
remote kernels and in exported notebooks, ~1.5 MB per view); `mode="server"` embeds an iframe to a local server (small notebooks,
but the browser must reach the kernel's localhost).

### Scene editor

`aevion edit --open` opens an empty scene. Pick a tool and click the ground to place buildings, trees, spheres, waypoints, base
stations and the start position; drag to move; `Delete` removes; edit dimensions in the properties panel. Download config
produces an Aevion configuration (`schema_version: 2`) with the world, start position, a waypoint trajectory and, if you placed
base stations, an `nr_cellular` network:

<!-- docs-test -->
```python
from aevion.viewer.editor import empty_payload, write_editor

page = write_editor("editor.html")
assert empty_payload()["meta"]["editor"] is True and page.exists()
```

```bash
aevion run scene.json --out out --open
```

`Save scene` / `Load scene…` keep an editable `.json` of the objects themselves. The configuration the editor writes is tested by
loading it with the normal loader and running the engine on it.

### The data format

A run is described by `aevion.view/2`:

* `meta`: title, vehicle count `n`, frames `T`, rotors `M`, vehicle spec, events, seed, config hash, `shapes` of every channel;
* channels: `t` (seconds from the first sample), `pos`, `quat` (w, x, y, z, body→world), `vel`, `euler`, `rotor`, `thrust`,
  `power`, `soc`, `wind`, `ref`, `err`, as float32 arrays of shape `(T, N, k)`; every other per-vehicle channel of the archive is
  an *extra* named `x:<name>` (`x:link_sinr_db`, …);
* layers: `world`, `lidar`, `wireless` (cells, outage threshold), `camera`, `planned`, `rem`, `beams`, `runs`.

Embedded in a page the arrays are base64; over HTTP and WebSocket they travel in a binary container
(`[u32 header length][JSON header][pad][float32 arrays]`, implemented in `aevion.viewer.protocol` and `web/src/data/decode.js` and
tested against each other). Frames are in the archive's world frame (ENU, z up; body FLU), so positions are drawn verbatim.

### Architecture

```text
viewer/
  export.py     single-file bundler (rewrites imports, blob modules + import map)     payload.py   aevion.view/2 builder
  server.py     HTTP API + static files + WebSocket upgrade                            protocol.py  binary frame container
  live.py       LiveHub, LiveStreamHook, serve_live                                    ws.py        RFC 6455 client/server
  video.py      frames/GIF/MP4 via headless browser                                    browser.py   DevTools Protocol client
  compare.py  editor.py  jupyter.py
  web/
    index.html  css/aevion.css  vendor/three/ (pinned, sha256 manifest)
    src/core/     util, playback clock, settings
    src/data/     decode, frames (store, interpolation, LOD), sources (static / chunked / live), derived, colormaps, editor model
    src/scene/    stage (renderer, cameras, shadows), sky, terrain, world
    src/vehicles/ procedural, glTF loader, fleet (detailed ≤ 16, instanced beyond)
    src/layers/   one file per overlay (trails, wireless, lidar, coverage, beams, ghosts, ...)
    src/ui/       hud, charts, timeline, panels, compare, editor, help
    tests/        node --test (pure logic: decoding, interpolation, decimation, editor model)
```

There is no build step and no TypeScript: the app is plain ES modules loaded through an import map, which keeps the shipped files
identical to the sources and lets the single-file export be a mechanical rewrite. Layers are isolated: one that throws is switched
off with a message instead of blanking the viewer.

### Performance and limits

Measured on a development laptop with *software* rendering (SwiftShader, no GPU), at 1280×720: a 200-vehicle replay with trails,
a wireless layer and five synchronised charts takes about 60 ms per frame; 100+ vehicles are drawn with instancing (three draw
calls for all bodies). Real GPUs are much faster, but the project does *not* yet claim a figure for the roadmap target (one hour,
50 vehicles at 30 fps on integrated graphics): that needs measuring on such hardware. The renderer lowers its own quality when frames
stay slow (`Quality: Auto`).

Known limits: the live replay buffer is bounded (see above); the coverage map is a deterministic large-scale map (median path loss,
no shadowing or fading) of the network as configured; antenna lobes show the pattern, not the beamforming weights; comparison runs
embedded in a single file are decimated to 1200 frames each; glTF models are not animated; the browser tests drive Chrome through the
DevTools Protocol rather than Playwright, and are skipped when no Chromium-family browser is installed.

---

<!-- page: archives -->

## Run archives

### Formats

| format | extension | notes |
|---|---|---|
| v2 store | `.aevd` (directory) | chunked, streamable, lazy, hashed. Default. |
| packed v2 | `.aevz` (one ZIP file) | read in place (no unpacking needed) for sharing |
| v1 file | `.aev` (compressed NPZ) | the legacy format; still read and written |

`RunArchive.load(path)` detects the format. `aevion convert a.aev b.aevd` converts by output extension.

### v2 layout

```text
run.aevd/
  manifest.json
  channels/pos/000000.npy  000001.npy …       one NumPy file per chunk of rows (time steps)
  channels/sensors__imu/…
```

`manifest.json` holds `schema` (`aevion.run/2`), `status` (`complete` | `incomplete`), `meta` (configuration, vehicle,
world, events, kernel, …), `provenance`, and for every channel its dtype, row shape, length and the list of chunks with
BLAKE2b hashes.

* Bounded memory: `ArchiveWriter` buffers `chunk_steps` rows per channel; `logging.mode: stream` writes while the
  simulation runs, so a 10-hour run needs the same RAM as a 10-second run.
* Crash tolerance: the manifest is replaced atomically after every flush. A killed run opens as
  `status: incomplete` up to its last flush. A run that fails with an exception (e.g. `SimulationDiverged`) is closed as
  `diverged`/`failed` with the error message, and remains readable.
* Lazy reads: `archive["pos"]` materialises a channel (cached); `archive.lazy("pos")[1000:1200, 0]` reads only the
  chunks it needs; `iter_chunks()` streams. `archive.release()` drops caches; `slice_time(t0, t1)` returns an in-memory
  sub-archive.
* Integrity: `aevion check run.aevd` re-hashes every chunk and reports missing, truncated or corrupt files.
  `--quick` skips re-hashing.

### Provenance

Stored in every archive (`archive.provenance`): Aevion/Python/library versions; a hash of the installed package set;
the git commit of the Aevion source and whether the tree was dirty; CPU count and Numba threads; the **configuration
hash**, seed, dynamics kernel and backend; loaded plugins. It deliberately omits user name, host name, paths and
environment variables, so archives can be shared without leaking personal information.

### Comparing and tracking runs

```text
aevion diff a.aevd b.aevd          # config changes, per-channel max|Δ| and first divergence time, metric deltas
aevion runs list --tag baseline    # SQLite registry at ~/.aevion/runs.db  ($AEVION_HOME to relocate)
aevion runs add run.aevd --tags baseline
aevion runs find <config-hash>     # every run that used this exact configuration
```

### Export

`archive.to_csv(path, vehicle)`, `archive.to_hdf5(path)`, `aevion.datasets.to_parquet(archive, path)` (long or per-vehicle
table; manifest in the schema metadata), `to_zarr(archive, path)` (needs `zarr`). Parquet export needs `pyarrow`; on
some Windows/Python-3.14 environments `pyarrow`'s native extension is unstable, which is why `aevion info` never
imports it.

---

<!-- page: plugins -->

## Writing a plugin

> Tip: Start from the template
>
> `aevion new-plugin my-aevion-ext` generates an installable package with an example sensor, a hook, the entry point
> and a passing test-suite. Edit it instead of starting from an empty directory.

A plugin is an ordinary Python package that registers components with Aevion. After installation they can be used
by name in any configuration file, listed by `aevion registry`, and validated like built-ins.

### 1. Write the component

```python
# my_aevion_ext/sonar.py
from aevion.sensors.base import Sensor

class Sonar(Sensor):
    """Downward sonar with a hard range limit"""
    name = "sonar"

    def __init__(self, rate=20.0, max_range=5.0, noise_std=0.02, **kw):   # **kw forwards rate/latency/dropout/name/clock/extrinsics
        super().__init__(rate=rate, **kw)
        self.max_range, self.noise_std = max_range, noise_std

    def measure(self, state):
        h = state.pos[:, 2]
        r = self.B.minimum(h + self.noise(h.shape, self.noise_std), self.max_range * self.B.ones(1))
        return dict(range=r, t=state.t)
```

### 2. Register it

```python
# my_aevion_ext/plugin.py
def register(registries):
    from .sonar import Sonar
    registries.sensors.register("sonar", Sonar, source="my_aevion_ext")
```

### 3. Declare the entry point

```toml
[project.entry-points."aevion.plugins"]
my_aevion_ext = "my_aevion_ext.plugin:register"
```

`pip install -e .` and then:

```yaml
sensors:
  - imu
  - {type: sonar, max_range: 8.0, noise_std: 0.05}
```

```text
$ aevion registry sensors        # shows 'sonar  <my_aevion_ext>'
$ aevion plugins                 # ok   my_aevion_ext  0.1.0
```

### What each registry expects

| registry | factory signature | returns |
|---|---|---|
| `controllers` | `f(sim, **params)` | object with `reset(n)` and `act(state, ref, dt) -> rotor commands` |
| `trajectories` | `f(**params)` | `Trajectory` (`sample(t)`, `reference(B, t, n, offsets)`) |
| `sensors` | `f(**params)` | `Sensor` subclass instance |
| `estimators` | `f(sim, **params)` | object with `reset(state)` and `update(readings, fresh)` |
| `winds` | `f(**params)` | `Wind` (`reset`, `sample(B, pos, vel, t, dt)`) |
| `worlds` | `f(seed=0, **params)` | `World` |
| `networks` | `f(world=None, seed=0, **params)` | object with `evaluate(pos)` and `to_dict()` |
| `formations` | `f(n, **params)` | `(n, 3)` offsets |
| `faults` | `f(**params)` | `Fault` (`on_start`, `on_end`, `modify_command`) |
| `hooks` | `f(**params)` | `Hook` |
| `health_checks` | `f(**params)` | `HealthCheck` |

Entries may carry default parameters that sit below whatever the user sets:
`registries.sensors.register("lidar16", Lidar, defaults={"channels": 16, "samples": 360})`.

### Failure isolation

If a plugin fails to import or its `register` raises, Aevion emits a warning, records the failure (`aevion plugins`
shows the error) and carries on; other plugins and the built-ins are unaffected.

A complete, runnable example lives in `examples/plugin_example/`.

### Trust

Plugins run with your privileges: install only plugins you trust. `AEVION_DISABLE_PLUGINS=1` skips plugin loading entirely, and
`aevion run --untrusted` refuses configurations that use plugin components (see [untrusted configurations](#untrusted-configurations)).

---

<!-- page: untrusted -->

## Untrusted configurations

A configuration decides how much CPU, memory and disk a run uses and which code it can reach. If you run
configurations written by other people (a web service, a grading system, a shared cluster front end), apply
resource limits before anything is built.

### From the command line

```bash
aevion validate --untrusted submitted.yaml       # structural checks + limits, no simulation
aevion run --untrusted submitted.yaml --out results/
```

`--untrusted` uses the conservative `ResourceLimits.untrusted()` preset: files larger than 256 KB, nesting deeper than
24 levels, more than 20,000 nodes or any YAML alias are refused; duration, steps, vehicles, vehicle-steps, sensor ray
throughput, camera resolution, world size and estimated archive size are capped; output is confined to `--out`;
only built-in components are allowed (no plugin components, no custom hooks); and the run is aborted after 300 s of
wall-clock time.

### From Python

<!-- docs-test -->
```python
from aevion.safety import ResourceLimitError, ResourceLimits, run_limited

limits = ResourceLimits.untrusted()           # or ResourceLimits(max_duration_s=60, max_vehicles=8, ...)
archive = run_limited({"duration": 2.0, "sensors": ["imu"]}, limits)
print(archive.n_steps, "steps")

try:
    run_limited({"duration": 100000.0}, limits)
except ResourceLimitError as e:
    print(e)                                    # lists *every* violated limit
```

Use `estimate_cost(config)` to see the model the checks are based on, and `check_config(config, limits)` to validate
without running.

### What the limits are, and are not

They bound resource use: parser bombs (size, depth, alias expansion), simulation cost, sensor throughput, output size
and location, run time. They are not a sandbox: plugins, custom hooks and any Python you pass in are trusted code.
For hostile code use OS-level isolation (containers, VMs, unprivileged users) and set `AEVION_DISABLE_PLUGINS=1`.

Archives are untrusted input too. Opening one validates its manifest (numeric dtypes only, bounded shapes, safe
relative chunk paths) and never unpickles anything; `aevion unpack` rejects path traversal, symlinks and zip bombs.
See [SECURITY](#security-policy) for the full threat model.

---

<!-- page: config-reference -->

<!-- config-reference:start -->
## Configuration reference (schema version 2)

*Generated from the code by `tools/gen_config_docs.py`: do not edit by hand.*

Write `schema_version: 2` at the top of hand-written files. Components accept a bare name (`controller: geometric`) or a mapping with a 
`type` (alias `kind`) key plus parameters. Machine-readable version: [simulation.schema.json](docs/simulation.schema.json).

#### `SimulationConfig`

Everything needed to reproduce a simulation.

| field | type | default | constraints | description |
|---|---|---|---|---|
| `schema_version` | an integer | `2` |  |  |
| `name` | a string | `"simulation"` |  |  |
| `description` | a string | `""` |  |  |
| `tags` | a list of a string | `[]` |  |  |
| `seed` | an integer | `0` | ≥ 0 |  |
| `duration` | a number | `10.0` | > 0.0 |  [s] |
| `engine` | a EngineConfig mapping | *EngineConfig* |  |  |
| `world` | a WorldConfig mapping | *WorldConfig* |  |  |
| `wind` | a WindConfig mapping | *WindConfig* |  |  |
| `vehicle` | a VehicleConfig mapping | *VehicleConfig* |  |  |
| `controller` | a ControllerConfig mapping | *ControllerConfig* |  |  |
| `trajectory` | a TrajectoryConfig mapping | *TrajectoryConfig* |  |  |
| `sensors` | a list of a SensorConfig mapping | `[]` |  |  |
| `estimator` | a EstimatorConfig mapping or null | `null` |  |  |
| `network` | a NetworkConfig mapping or null | `null` |  |  |
| `swarm` | a SwarmConfig mapping or null | `null` |  |  |
| `faults` | a list of a FaultConfig mapping | `[]` |  |  |
| `hooks` | a list of a HookConfig mapping | `[]` |  |  |
| `health` | a HealthConfig mapping | *HealthConfig* |  |  |
| `environment` | a EnvironmentConfig mapping | *EnvironmentConfig* |  |  |
| `logging` | a LoggingConfig mapping | *LoggingConfig* |  |  |

#### `EngineConfig`

EngineConfig(dt: 'float' = 0.005, integrator: "Literal['euler', 'semi_implicit', 'midpoint', 'rk4']" = 'rk4', backend: 'str' = 'numpy', dtype: "Optional[Literal['float32', 'float64']]" = None, kernel: "Literal['auto', 'reference', 'numba']" = 'auto', verify_kernel: 'bool' = True, controller_rate_hz: 'Optional[float]' = None)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `dt` | a number | `0.005` | > 0.0; ≤ 0.05 | physics step [s] |
| `integrator` | one of 'euler', 'semi_implicit', 'midpoint', 'rk4' | `"rk4"` |  |  |
| `backend` | a string | `"numpy"` |  |  |
| `dtype` | one of 'float32', 'float64' or null | `null` |  |  |
| `kernel` | one of 'auto', 'reference', 'numba' | `"auto"` |  |  |
| `verify_kernel` | true/false | `true` |  |  |
| `controller_rate_hz` | a number or null | `null` | > 0.0 | controller rate; default = physics rate [Hz] |

#### `WorldConfig`

A procedural world (``forest``, ``city``, …) or a full serialised world (``World.to_dict()``).

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"empty"` |  |  |
| `params` | a mapping | `{}` |  |  |

*Component type is chosen from the `worlds` registry (`aevion registry worlds`); remaining keys are the component's own parameters.*

#### `WindConfig`

WindConfig(type: 'str' = 'none', params: 'Dict[str, Any]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"none"` |  |  |
| `params` | a mapping | `{}` |  |  |

*Component type is chosen from the `winds` registry (`aevion registry winds`); remaining keys are the component's own parameters.*

#### `VehicleConfig`

VehicleConfig(model: 'Union[str, Dict[str, Any]]' = 'generic_1kg', count: 'int' = 1, formation: 'Optional[FormationConfig]' = None, overrides: 'Dict[str, Any]' = <factory>, initial: 'InitialConfig' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `model` | a string or a mapping | `"generic_1kg"` |  |  |
| `count` | an integer | `1` | ≥ 1 |  |
| `formation` | a FormationConfig mapping or null | `null` |  |  |
| `overrides` | a mapping | `{}` |  |  |
| `initial` | a InitialConfig mapping | *InitialConfig* |  |  |

#### `InitialConfig`

InitialConfig(pos: 'Optional[Tuple[float, float, float]]' = None, vel: 'Optional[Tuple[float, float, float]]' = None, euler: 'Optional[Tuple[float, float, float]]' = None, omega: 'Optional[Tuple[float, float, float]]' = None, soc: 'float' = 1.0)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `pos` | a list or null | `null` |  |  |
| `vel` | a list or null | `null` |  |  |
| `euler` | a list or null | `null` |  | roll, pitch, yaw [rad] |
| `omega` | a list or null | `null` |  |  |
| `soc` | a number | `1.0` | ≥ 0.0; ≤ 1.0 | initial battery state of charge |

#### `FormationConfig`

FormationConfig(type: 'str' = 'line', params: 'Dict[str, Any]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"line"` |  |  |
| `params` | a mapping | `{}` |  |  |

*Component type is chosen from the `formations` registry (`aevion registry formations`); remaining keys are the component's own parameters.*

#### `ControllerConfig`

ControllerConfig(type: 'str' = 'geometric', params: 'Dict[str, Any]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"geometric"` |  |  |
| `params` | a mapping | `{}` |  |  |

*Component type is chosen from the `controllers` registry (`aevion registry controllers`); remaining keys are the component's own parameters.*

#### `TrajectoryConfig`

TrajectoryConfig(type: 'str' = 'hover', params: 'Dict[str, Any]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"hover"` |  |  |
| `params` | a mapping | `{}` |  |  |

*Component type is chosen from the `trajectories` registry (`aevion registry trajectories`); remaining keys are the component's own parameters.*

#### `SensorConfig`

SensorConfig(type: 'str' = '', params: 'Dict[str, Any]' = <factory>, name: 'Optional[str]' = None, record: 'bool' = False)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `""` |  |  |
| `params` | a mapping | `{}` |  |  |
| `name` | a string or null | `null` |  |  |
| `record` | true/false | `false` |  |  |

*Component type is chosen from the `sensors` registry (`aevion registry sensors`); remaining keys are the component's own parameters.*

#### `EstimatorConfig`

EstimatorConfig(type: 'str' = 'mahony_kf', params: 'Dict[str, Any]' = <factory>, handover_s: 'float' = 6.0)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"mahony_kf"` |  |  |
| `params` | a mapping | `{}` |  |  |
| `handover_s` | a number | `6.0` | ≥ 0.0 | fly the controller on truth until the filter has converged [s] |

*Component type is chosen from the `estimators` registry (`aevion registry estimators`); remaining keys are the component's own parameters.*

#### `NetworkConfig`

NetworkConfig(type: 'str' = 'cellular', params: 'Dict[str, Any]' = <factory>, rate_hz: 'Optional[float]' = None)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `"cellular"` |  |  |
| `params` | a mapping | `{}` |  |  |
| `rate_hz` | a number or null | `null` | > 0 | evaluation rate; default = recording rate [Hz] |

*Component type is chosen from the `networks` registry (`aevion registry networks`); remaining keys are the component's own parameters.*

#### `SwarmConfig`

SwarmConfig(avoid_radius: 'float' = 1.0, gain: 'float' = 0.8, max_push: 'float' = 1.5, comm: 'CommModelConfig' = <factory>, rate_hz: 'Optional[float]' = None, laws: 'List[SwarmLawConfig]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `avoid_radius` | a number | `1.0` | > 0.0 |  [m] |
| `gain` | a number | `0.8` | ≥ 0.0 |  |
| `max_push` | a number | `1.5` | > 0.0 |  [m] |
| `comm` | a CommModelConfig mapping | *CommModelConfig* |  |  |
| `rate_hz` | a number or null | `null` | > 0.0 | coordination rate; default = every physics step [Hz] |
| `laws` | a list of a SwarmLawConfig mapping | `[]` |  |  |

#### `CommModelConfig`

CommModelConfig(delay_s: 'float' = 0.0, loss: 'float' = 0.0, range_m: 'float' = inf, rate_hz: 'float' = 20.0)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `delay_s` | a number | `0.0` | ≥ 0.0 |  [s] |
| `loss` | a number | `0.0` | ≥ 0.0; ≤ 1.0 | packet-loss probability per link |
| `range_m` | a number | `inf` | > 0.0 |  [m] |
| `rate_hz` | a number | `20.0` | > 0.0 |  [Hz] |

#### `FaultConfig`

A fault active during ``[t_start, t_end)`` (``t_end`` omitted = permanent).

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `""` |  |  |
| `params` | a mapping | `{}` |  |  |
| `t_start` | a number | `0.0` | ≥ 0.0 |  [s] |
| `t_end` | a number or null | `null` | ≥ 0.0 |  [s] |
| `vehicle` | an integer or null | `null` | ≥ 0 | vehicle index (omit = all) |

*Component type is chosen from the `faults` registry (`aevion registry faults`); remaining keys are the component's own parameters.*

#### `HookConfig`

HookConfig(type: 'str' = '', params: 'Dict[str, Any]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `type` | a string | `""` |  |  |
| `params` | a mapping | `{}` |  |  |

*Component type is chosen from the `hooks` registry (`aevion registry hooks`); remaining keys are the component's own parameters.*

#### `HealthConfig`

HealthConfig(enabled: 'bool' = True, on_diverge: "Literal['raise', 'stop', 'warn']" = 'raise', check_every: 'int' = 1, max_speed: 'float' = 200.0, max_rate: 'float' = 500.0, max_altitude: 'float' = 20000.0, quat_tol: 'float' = 1e-06, checks: 'List[ComponentConfig]' = <factory>)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `enabled` | true/false | `true` |  |  |
| `on_diverge` | one of 'raise', 'stop', 'warn' | `"raise"` |  |  |
| `check_every` | an integer | `1` | ≥ 1 | run checks every N physics steps |
| `max_speed` | a number | `200.0` | > 0.0 |  [m/s] |
| `max_rate` | a number | `500.0` | > 0.0 | maximum body rate [rad/s] |
| `max_altitude` | a number | `20000.0` | > 0.0 |  [m] |
| `quat_tol` | a number | `1e-06` | > 0.0 |  |
| `checks` | a list of a ComponentConfig mapping | `[]` |  |  |

#### `LoggingConfig`

LoggingConfig(record_every: 'int' = 4, rate_hz: 'Optional[float]' = None, mode: "Literal['memory', 'stream']" = 'memory', path: 'Optional[str]' = None, chunk_steps: 'int' = 512, sensors: 'List[str]' = <factory>, channels: 'Optional[List[str]]' = None, vehicles: 'Optional[List[int]]' = None, provenance: 'bool' = True)

| field | type | default | constraints | description |
|---|---|---|---|---|
| `record_every` | an integer | `4` | ≥ 1 | record every N physics steps |
| `rate_hz` | a number or null | `null` | > 0.0 | record rate; overrides record_every [Hz] |
| `mode` | one of 'memory', 'stream' | `"memory"` |  |  |
| `path` | a string or null | `null` |  |  |
| `chunk_steps` | an integer | `512` | ≥ 16 |  |
| `sensors` | a list of a string | `[]` |  |  |
| `channels` | a list of a string or null | `null` |  |  |
| `vehicles` | a list of an integer or null | `null` |  |  |
| `provenance` | true/false | `true` |  |  |

### Registered components

```text
vehicles (9):
  camera_2kg    ≈2 kg camera drone (illustrative parameters, not an exact DJI model).
  crazyflie     Bitcraze Crazyflie 2.x (27 g). Parameters approximate those of Förster (2015)
  generic_1kg   Generic 1 kg research quadrotor, T/W≈2.2. Good default for control studies.
  hex_3kg       
  hexa          generic hexacopter
  octo          generic octocopter
  quad_plus     generic +-configuration quadrotor
  quad_x        generic X-configuration quadrotor (set arm, mass, kT, ...)
  racing_5inch  250 mm class FPV racing quad (≈0.6 kg, thrust-to-weight ≈ 6).

controllers (8):
  cascaded  [cascaded_pid, pid]  cascaded PID (position, velocity, attitude, rate), PX4-style
  geometric [geometric_se3, se3]  Geometric SE(3) tracking controller (Lee et al. 2010)
  indi       incremental nonlinear dynamic inversion torque law (inertia-model free)
  l1        [l1_adaptive]  L1 adaptive augmentation of the position loop
  lqr       [lqr_hover]  LQR about hover with acceleration feedforward
  mpc       [linear_mpc]  constrained linear MPC with a Kalman disturbance estimator (offset-free)
  mppi       sampling-based model-predictive path-integral control (obstacle-aware)
  smc       [sliding_mode]  sliding-mode translational loop with a boundary layer

trajectories (8):
  circle     Horizontal circle of radius ``r`` at angular rate ``omega`` [rad/s] (ramped in over ``ramp`` s).
  figure8   [figure_8, lemniscate]  Lissajous figure-eight in the horizontal plane
  hover     [hold]  Base class: ``sample(t)`` returns NumPy arrays; :meth:`reference` makes a batched Reference.
  lissajous  Smooth 3-D Lissajous curve; ``figure8()`` is the 1:2 case in the horizontal plane.
  mission   [behavior_tree]  behaviour-tree mission (takeoff, goto, hover, land, conditions)
  planned    collision-free minimum-snap trajectory planned around the world's obstacles
  polyline   constant-speed straight segments
  waypoints [min_snap]  minimum-snap-style polynomial through waypoints

sensors (14):
  baro         [barometer]  static-pressure altimeter
  camera        pinhole depth/RGB/segmentation camera (RL-sized 64 x 48 default)
  event_camera [dvs]  event camera: log-intensity change events with threshold mismatch, refractory period and background noise
  gnss          pseudorange GNSS receiver: constellations, atmosphere, multipath, NLoS, least-squares fix, DOP
  gps           GNSS receiver (Gauss-Markov error, latency)
  imu           6-axis MEMS IMU
  lidar         spinning multi-beam LiDAR (config default: 8 x 180 beams, no point cloud)
  mag          [magnetometer]  3-axis magnetometer
  optical_flow [flow]  downward optical-flow sensor (raw and gyro-compensated flow, quality)
  pitot         differential-pressure airspeed sensor (relative air velocity, lag, acceptance cone)
  radar         FMCW radar: radar-equation SNR, range/Doppler resolution, velocity ambiguity, optional range-Doppler map
  rangefinder   single-beam range sensor
  ultrasonic    cone-beam ultrasonic rangefinder (nearest echo, oblique-surface loss, speed-of-sound bias)
  uwb           UWB two-way ranging to fixed anchors (NLoS excess delay, outliers)

estimators (2):
  eskf      [error_state_kf]  15-state error-state Kalman filter (IMU + GNSS, baro, mag, range, flow, pose) with delayed fusion
  mahony_kf [default]  Mahony attitude filter + 9-state position/velocity Kalman filter

winds (10):
  composite       [compositewind]  sum of several wind models, given as specs
  constant        [constantwind]  Uniform wind. ``velocity`` is ``(3,)`` or per-env ``(N, 3)``.
  dryden          [drydenturbulence]  Dryden turbulence (MIL-F-8785C)
  gridded         [cfd]  wind imported from a regular grid (.npz or .csv), trilinear in space, linear in time
  gust             discrete 1-cos gust
  log_law         [loglaw]  neutral boundary-layer log profile with roughness length and displacement height
  none            [nowind, calm]  
  random_constant [randomconstantwind]  per-environment random constant wind (domain randomisation)
  shear           [windshear]  atmospheric boundary-layer profile
  urban            log-law wind modified around the world's buildings (cavity, wake, stagnation, downwash)

worlds (7):
  city       Manhattan-style building grid
  corridor   enclosed corridor with pillars
  empty      flat ground, no obstacles
  forest     random tree trunks
  from_dict  a serialised world (World.to_dict())
  hills      smooth random terrain
  scene     [geo_scene, mesh_scene]  declarative scene: origin, weather, terrain files, OSM/GeoJSON buildings, meshes and primitives

actors (5):
  bouncing     Moves at constant velocity and reflects off the faces of the axis-aligned box ``bounds = [x0, y0, z0, x1, y1, z1]`` (like a ball).
  circular     Horizontal circle of ``radius_path`` about ``center`` at angular rate ``omega`` [rad/s] (counter-clockwise for positive ``omega``).
  linear       Constant velocity from ``start``: ``p(t) = start + velocity * t``.
  random_walk  Smooth random motion: the velocity is an Ornstein-Uhlenbeck process (mean 0, std ``speed``, correlation time ``tau``).
  waypoints    Constant-speed travel along straight segments through ``points``; ``mode`` ``"loop"`` closes the path, ``"pingpong"`` reverses at the ends,

scan_patterns (4):
  lissajous    Azimuth ``A sin(2 pi fx t)``, elevation ``B sin(2 pi fy t + phase)`` over one scan; ``fx`` and ``fy`` are cycles per scan (integers repeat).
  mems_raster  ``lines`` scan lines of ``points`` samples: sinusoidal fast axis (azimuth, alternating direction) and a linear slow axis (elevation).
  rosette      Two Risley prisms: beam offset ``A1 e^{i w1 t} + A2 e^{i w2 t}`` (angles). ``fov_deg`` is the full circular field of view,
  spinning     Interface. ``n_rays`` beams per scan, ``static`` if every scan is identical, ``grid_shape`` ``(rows, cols)`` when the beams form a grid.

channels (1):
  air_to_ground  Al-Hourani air-to-ground channel (probabilistic or geometric LoS)

propagation (6):
  al_hourani    Large-scale model: median path loss for given LoS state, LoS probability and shadowing spread.
  free_space    Friis free-space loss: always line-of-sight, no shadowing. The upper bound on achievable link quality.
  log_distance  ``PL(d) = PL(d0) + 10 n log10(d / d0)`` with separate exponents for LoS and NLoS.
  tr36777       TR 36.777 (Annex B) aerial-vehicle models: ``uma_av``, ``umi_av``, ``rma_av``.
  tr38901       TR 38.901 path loss. ``scenario`` is one of ``uma``, ``umi``, ``rma``, ``inh``.
  two_ray       Two-ray ground-reflection model (perfectly reflecting flat ground, reflection coefficient -1).

antennas (7):
  gaussian_beam     Gaussian main lobe with independent azimuth/elevation half-power beamwidths and a sidelobe floor (relative to peak).
  half_wave_dipole  Half-wave dipole along z: ``D = 1.64 [cos(pi/2 sin(el)) / cos(el)]^2`` (2.15 dBi). The usual UAV whip/dipole.
  isotropic         Helper class that provides a standard way to create an ABC using
  parabolic         Aperture antenna: ``G = eta (pi D / lambda)^2``, half-power beamwidth ``70 lambda / D`` degrees, parabolic roll-off to a sidelobe floor.
  patch_3gpp        3GPP TR 38.901 Table 7.3-1 single-element pattern (also a good model of a directional patch/panel).
  sector_36814      TR 36.814 macro-cell sector (vertical tilt pattern only, as used for cellular base stations serving UAVs).
  short_dipole      Short (Hertzian) dipole along the antenna z axis: ``D = 1.5 cos^2(el)`` (1.76 dBi, omnidirectional in azimuth).

schedulers (3):
  max_ci             Serve the user with the best instantaneous rate (maximises sum rate, starves cell-edge users).
  proportional_fair  Serve ``argmax r_inst / r_avg^beta``; with ``beta = 1`` this maximises the sum of log-throughputs.
  round_robin        Helper class that provides a standard way to create an ABC using

networks (3):
  cellular     UAV as a cellular-connected user (base stations + air-to-ground channel)
  mesh         UAV-to-UAV mesh with multi-hop routing to a gateway
  nr_cellular  5G-NR-style cellular network: 3GPP path loss, antennas, fading, handover, RLF, scheduler, link adaptation

tasks (10):
  area_coverage         Visit every cell of a ``size × size`` metre area on a ``cells × cells`` grid; a cell is seen when the vehicle passes within ``footprint`` of its centre.
  connected_navigation  Fly to a goal **while staying connected**: communication-aware path planning as a learning problem.
  gate_racing           Fly through ``n_gates`` circular gates in order; a gate counts when the path crosses its plane in the travel direction within ``radius`` of its centre.
  hover                 Fly to a random point and hold it.
  inspection            Inspect points on the surface of a vertical cylinder: a point counts when the vehicle comes within ``sense`` of it; flying into the cylinder crashes.
  landing               Land on a platform that drifts across the ground: touchdown inside ``zone`` at a speed below ``max_touch_speed`` succeeds, anything else fails.
  navigation            Reach a goal through a world with obstacles using LiDAR observations.
  target_tracking       Stay with a target whose velocity is a bounded random process (unpredictable, unlike the scripted circle and figure-eight of ``tracking``).
  tracking              Track a moving reference (circle or figure-eight) with randomised size, rate and phase.
  waypoints             Visit ``n_waypoints`` random goals in sequence; a goal is reached within ``radius``.

formations (6):
  column  Single file along the direction of travel (x).
  grid    
  line    
  ring    Vehicles on a circle; neighbouring vehicles are ``spacing`` apart (chord length).
  sphere  Roughly even points on a sphere (Fibonacci lattice) whose nearest-neighbour distance is about ``spacing``.
  vee     V (arrow-head) formation, leader at the tip.

swarm_laws (3):
  coverage          [lloyd]  Lloyd coverage of a rectangle on the believed positions
  formation_keeping  neighbours wait for a lagging vehicle: the shape holds under disturbances
  rendezvous         average consensus on position over the lossy link

faults (24):
  actuator_stuck       Rotor ``rotor`` ignores the controller and stays at ``speed`` rad/s (``None`` = its speed when the fault begins).
  battery_drop         Instantly reduce the state of charge to at most ``soc`` (a cell failure / sudden brown-out).
  com_shift            Shift the centre of mass by ``offset`` [m] (body frame).
  comm_blackout        Total loss of the inter-agent link (``swarm``), the cellular link (``network``) or both.
  gnss_jamming         Interference source near the vehicle degrades the GNSS receiver.
  gnss_spoofing        Threat model: a counterfeit GNSS signal drags the reported position away from the truth.
  gps_outage           The sensor loses each sample with ``probability`` (1.0 = complete outage).
  network_jamming      An interference source degrades the cellular link: its power adds to the interference at the UAV (needs a network with ``jammers``).
  payload_change       Add (or release, if negative) a payload mass [kg].
  rotor_failure        Rotor ``rotor`` loses ``severity`` of its thrust (1.0 = dead; the commanded speed is also forced to zero).
  sensor_bias_step     Add a constant offset to one field of the sensor's reading (e.g. a gyro bias jump).
  sensor_delay         Add ``delay_s`` of latency (a congested bus, a processing stall).
  sensor_delay_spikes  Random extra latency: each sample is delayed by ``delay_s`` with probability ``probability``. Delivery stays in order (a late sample holds
  sensor_drift         A bias that grows linearly from the start of the fault: ``bias(t) = rate * (t - t_start)`` (per component, or a 3-vector ``rate``).
  sensor_dropout       The sensor loses each sample with ``probability`` (1.0 = complete outage).
  sensor_nan           With probability ``probability`` the field arrives as NaN (a corrupted frame, a bus error). Estimators must reject it.
  sensor_noise_scale   Multiply the sensor's noise by ``factor`` (degradation, vibration, interference).
  sensor_saturation    Clip a field to ``[-limit, limit]`` (a range that is too small, a railed ADC).
  sensor_scale         Multiply a field by ``factor`` (e.g. 1.05: a 5 % scale-factor error from temperature or a damaged sensor).
  sensor_spikes        Each sample of a vehicle is corrupted with probability ``probability`` by an outlier: Gaussian with standard deviation ``magnitude``
  sensor_stuck         The sensor keeps republishing its last reading.
  thrust_loss          Scale the thrust coefficient of all rotors of the target vehicle(s) (e.g. damaged propellers).
  wind_disturbance     Add a constant wind ``velocity`` [m/s] (ENU) on top of the existing wind for the fault's duration.
  wind_gust            A 1-cos gust of ``amplitude`` [m/s] lasting ``duration`` s, starting when the fault starts.

hooks (7):
  battery_model      detailed battery: Peukert capacity, polarisation, heating, ageing
  event_log          print events as they occur
  profiler           per-task wall-time report
  progress           print a progress bar
  proximity_effects  ceiling thrust gain and wall suction from nearby surfaces
  slung_load         a payload hanging on an elastic rope (tension, swing, slack)
  stop_on_event      stop the run when an event matches

health_checks (9):
  altitude         Base class. Subclasses implement :meth:`check` and set ``hard``.
  battery          Soft: warns once a battery is depleted.
  body_rate        Base class. Subclasses implement :meth:`check` and set ``hard``.
  finite           Every state array is finite. Uses a single summed probe first (cheap), details only on failure.
  geofence         geofences / no-fly zones: emits geofence.breach and geofence.clear events
  quaternion_norm  Base class. Subclasses implement :meth:`check` and set ``hard``.
  rotor_speed      Rotor speeds stay non-negative and below 1.25 x the nominal maximum.
  saturation       Soft: warns when any rotor is pinned at its limit for more than ``window_s`` seconds.
  speed            Base class. Subclasses implement :meth:`check` and set ``hard``.

metrics (1):
  flight  mean over vehicles of the standard flight metrics (tracking error, energy, tilt, ...)
```
<!-- config-reference:end -->

---

<!-- page: cli -->

## Command line

Generated from the argument parser (`aevion --help`).

```text
usage: aevion [-h]
              {info,vehicles,registry,plugins,validate,schema,new-plugin,init,demo,run,replay,live,edit,report,inspect,verify,check,diff,convert,pack,unpack,runs,bench,experiment,suite,vision-dataset,methods} ...

Aevion: UAV simulation for research

positional arguments:
  {info,vehicles,registry,plugins,validate,schema,new-plugin,init,demo,run,replay,live,edit,report,inspect,verify,check,diff,convert,pack,unpack,runs,bench,experiment,suite,vision-dataset,methods}
    info                versions, accelerators and registered components
    vehicles            list vehicle presets
    registry            list registered components (controllers, sensors,
                        winds, faults, …)
    plugins             list installed plugins
    validate            validate configuration files and report every problem
    schema              print/write the JSON Schema of the configuration
                        format
    new-plugin          scaffold an installable plugin package (sensor + hook
                        + tests)
    init                write a starter configuration file
    demo                run a demo flight, then write archive + report + 3-D
                        replay
    run                 run a configuration file (YAML/JSON; schema v2 or legacy
                        scenario)
    replay              interactive 3-D replay of an archive: HTML file, local
                        server, comparison, screenshot or video
    live                run a configuration and watch it live in the browser
    edit                write the in-browser scene editor (exports a
                        configuration for `aevion run`)
    report              figures, tables and insights for an archive
    inspect             summary and metrics of an archive
    verify              re-simulate an archive from its configuration and
                        check it reproduces
    check               verify the integrity (hashes) of a v2 archive
    diff                compare two archives (configuration, channels,
                        metrics)
    convert             convert an archive (.aev ⇄ .aevd ⇄ .aevz) by output
                        extension
    pack                pack a .aevd directory into one shareable .aevz file
    unpack              unpack a .aevz file
    runs                manage the run registry
    bench               benchmark simulation throughput on this machine
    experiment          run an experiment specification (grid of simulation
                        configurations) with caching, sharding and a result
                        database
    suite               benchmark suites: compare controllers on fixed
                        scenarios and rank them
    vision-dataset      fly a configuration with a camera and write a labelled
                        image dataset (PNG, COCO, YOLO, depth, flow)
    methods             print an auto-generated methods appendix for a
                        configuration

options:
  -h, --help            show this help message and exit
```

### `aevion info`

```text
usage: aevion info [-h]

options:
  -h, --help  show this help message and exit
```

### `aevion vehicles`

```text
usage: aevion vehicles [-h]

options:
  -h, --help  show this help message and exit
```

### `aevion registry`

```text
usage: aevion registry [-h] [kind]

positional arguments:
  kind

options:
  -h, --help  show this help message and exit
```

### `aevion plugins`

```text
usage: aevion plugins [-h]

options:
  -h, --help  show this help message and exit
```

### `aevion validate`

```text
usage: aevion validate [-h] [--untrusted] files [files ...]

positional arguments:
  files

options:
  -h, --help   show this help message and exit
  --untrusted  also enforce the strict untrusted-configuration resource limits
```

### `aevion schema`

```text
usage: aevion schema [-h] [-o OUT]

options:
  -h, --help     show this help message and exit
  -o, --out OUT
```

### `aevion new-plugin`

```text
usage: aevion new-plugin [-h] [--dest DEST] [--description DESCRIPTION]
                         [--force]
                         name

positional arguments:
  name                  distribution name, e.g. my-aevion-ext

options:
  -h, --help            show this help message and exit
  --dest DEST           output directory (default: ./<name>)
  --description DESCRIPTION
  --force               write into an existing non-empty directory
```

### `aevion init`

```text
usage: aevion init [-h] [--name NAME] [out]

positional arguments:
  out

options:
  -h, --help   show this help message and exit
  --name NAME
```

### `aevion demo`

```text
usage: aevion demo [-h] [--vehicle VEHICLE]
                   [--controller {geometric,cascaded,lqr}]
                   [--trajectory {hover,circle,figure8,waypoints}]
                   [--world {empty,forest,city,corridor,hills}] [--wind WIND]
                   [--duration DURATION] [-n N] [--seed SEED] [--name NAME]
                   [--out OUT] [--no-report] [--offline] [--open] [--register]
                   [--format {v1,v2}]

options:
  -h, --help            show this help message and exit
  --vehicle VEHICLE
  --controller {geometric,cascaded,lqr}
  --trajectory {hover,circle,figure8,waypoints}
  --world {empty,forest,city,corridor,hills}
  --wind WIND           mean wind speed [m/s] (0 = calm)
  --duration DURATION
  -n N                  number of vehicles
  --seed SEED
  --name NAME
  --out OUT
  --no-report
  --offline             accepted for compatibility; the replay is always self-
                        contained
  --open                open the replay in a browser
  --register            add the run to the run registry (~/.aevion/runs.db)
  --format {v1,v2}      archive format (v2 = chunked store)
```

### `aevion run`

```text
usage: aevion run [-h] [--out OUT] [--backend BACKEND] [--stream] [--profile]
                  [--untrusted] [--no-report] [--offline] [--open]
                  [--register] [--format {v1,v2}]
                  scenario

positional arguments:
  scenario

options:
  -h, --help         show this help message and exit
  --out OUT
  --backend BACKEND
  --stream           stream the archive to disk while running (bounded memory)
  --profile          print per-task timing
  --untrusted        strict resource limits and built-in components only (for
                     configs from unknown sources)
  --no-report
  --offline          accepted for compatibility; the replay is always self-
                     contained
  --open             open the replay in a browser
  --register         add the run to the run registry (~/.aevion/runs.db)
  --format {v1,v2}   archive format (v2 = chunked store)
```

### `aevion replay`

```text
usage: aevion replay [-h] [--out OUT] [--offline] [--open] [--serve]
                     [--host HOST] [--port PORT]
                     [--compare ARCHIVE [ARCHIVE ...]] [--label LABEL]
                     [--theme {dark,light,print}] [--screenshot PNG] [--t T]
                     [--no-ui] [--video FILE] [--fps FPS] [--speed SPEED]
                     [--sweep SWEEP] [--camera {orbit,chase,top,fpv}]
                     [--size SIZE] [--ui]
                     archive

positional arguments:
  archive

options:
  -h, --help            show this help message and exit
  --out OUT             HTML output path (default: next to the archive)
  --offline             accepted for compatibility; the replay is always self-
                        contained
  --open                open the result in a browser
  --serve               serve it from a local web server instead of writing a
                        file (best for long runs)
  --host HOST
  --port PORT
  --compare ARCHIVE [ARCHIVE ...]
                        more runs shown as ghosts next to the first
  --label LABEL         name of each run (repeat; main run first)
  --theme {dark,light,print}
  --screenshot PNG      render one frame to a PNG (needs Chrome/Edge/Chromium)
  --t T                 time of the screenshot in seconds (default: middle of
                        the run)
  --no-ui               screenshot without the charts and panels
  --video FILE          .mp4/.webm (ffmpeg), .gif (Pillow) or a directory for
                        PNG frames
  --fps FPS
  --speed SPEED         simulated seconds per video second
  --sweep SWEEP         turn the orbit camera by this many degrees over the
                        clip
  --camera {orbit,chase,top,fpv}
  --size SIZE
  --ui                  include charts and panels in the video
```

### `aevion live`

```text
usage: aevion live [-h] [--speed SPEED] [--host HOST] [--port PORT] [--open]
                   [--no-wait] [--keep] [--theme {dark,light,print}]
                   [--save ARCHIVE]
                   scenario

positional arguments:
  scenario

options:
  -h, --help            show this help message and exit
  --speed SPEED         simulated seconds per wall second (0 = as fast as
                        possible)
  --host HOST
  --port PORT
  --open                open a browser
  --no-wait             start immediately instead of waiting for a browser to
                        connect
  --keep                keep serving after the run ends
  --theme {dark,light,print}
  --save ARCHIVE        also save the finished run
```

### `aevion edit`

```text
usage: aevion edit [-h] [--out OUT] [--open]

options:
  -h, --help  show this help message and exit
  --out OUT   HTML path (default: scene_editor.html)
  --open
```

### `aevion report`

```text
usage: aevion report [-h] [--out OUT] [--offline] archive

positional arguments:
  archive

options:
  -h, --help  show this help message and exit
  --out OUT
  --offline
```

### `aevion inspect`

```text
usage: aevion inspect [-h] archive

positional arguments:
  archive

options:
  -h, --help  show this help message and exit
```

### `aevion verify`

```text
usage: aevion verify [-h] [--tol TOL] archive

positional arguments:
  archive

options:
  -h, --help  show this help message and exit
  --tol TOL
```

### `aevion check`

```text
usage: aevion check [-h] [--quick] archive

positional arguments:
  archive

options:
  -h, --help  show this help message and exit
  --quick     skip re-hashing chunk data
```

### `aevion diff`

```text
usage: aevion diff [-h] [--tol TOL] [--strict] a b

positional arguments:
  a
  b

options:
  -h, --help  show this help message and exit
  --tol TOL
  --strict    also fail if the configurations differ
```

### `aevion convert`

```text
usage: aevion convert [-h] [--chunk CHUNK] src dst

positional arguments:
  src
  dst

options:
  -h, --help     show this help message and exit
  --chunk CHUNK
```

### `aevion pack`

```text
usage: aevion pack [-h] [--out OUT] archive

positional arguments:
  archive

options:
  -h, --help  show this help message and exit
  --out OUT
```

### `aevion unpack`

```text
usage: aevion unpack [-h] [--out OUT] archive

positional arguments:
  archive

options:
  -h, --help  show this help message and exit
  --out OUT
```

### `aevion runs`

```text
usage: aevion runs [-h] [--name NAME] [--tag TAG] [--tags [TAGS ...]]
                   [--metric METRIC] [--limit LIMIT] [--remove]
                   [--delete-files]
                   {list,add,show,tag,rm,prune,find} [target]

positional arguments:
  {list,add,show,tag,rm,prune,find}
  target

options:
  -h, --help            show this help message and exit
  --name NAME
  --tag TAG
  --tags [TAGS ...]
  --metric METRIC
  --limit LIMIT
  --remove
  --delete-files
```

### `aevion bench`

```text
usage: aevion bench [-h] [-n N] [--steps STEPS]

options:
  -h, --help     show this help message and exit
  -n N
  --steps STEPS
```

### `aevion experiment`

```text
usage: aevion experiment [-h] [--cache CACHE] [--db DB] [--workers WORKERS]
                         [--shard SHARD] [--of OF] [--metric METRIC]
                         [--csv CSV] [--shards SHARDS] [--partition PARTITION]
                         [--time TIME] [--mem MEM] [--cpus CPUS] [-o OUT]
                         {run,slurm} spec

positional arguments:
  {run,slurm}           run it here, or write a Slurm job-array script
  spec                  experiment JSON written by ConfigExperiment.save_spec

options:
  -h, --help            show this help message and exit
  --cache CACHE         result cache directory (makes runs resumable)
  --db DB               SQLite result database to store the rows in
  --workers WORKERS
  --shard SHARD
  --of OF               number of shards (this process runs shard --shard)
  --metric METRIC       metric to print
  --csv CSV             write the rows as CSV
  --shards SHARDS       (slurm) number of array tasks
  --partition PARTITION
  --time TIME
  --mem MEM
  --cpus CPUS
  -o, --out OUT         (slurm) write the script here
```

### `aevion suite`

```text
usage: aevion suite [-h] [--controllers CONTROLLERS [CONTROLLERS ...]]
                    [--cache CACHE] [--db DB] [--workers WORKERS] [--out OUT]
                    {list,run} [name]

positional arguments:
  {list,run}
  name

options:
  -h, --help            show this help message and exit
  --controllers CONTROLLERS [CONTROLLERS ...]
                        registered controller names to compare
  --cache CACHE
  --db DB
  --workers WORKERS
  --out OUT             write the leaderboard JSON
```

### `aevion vision-dataset`

```text
usage: aevion vision-dataset [-h] --out OUT [--camera CAMERA] [--every EVERY]
                             [--max-frames MAX_FRAMES]
                             config

positional arguments:
  config

options:
  -h, --help            show this help message and exit
  --out OUT
  --camera CAMERA
  --every EVERY
  --max-frames MAX_FRAMES
```

### `aevion methods`

```text
usage: aevion methods [-h] [--format {markdown,latex}] config

positional arguments:
  config

options:
  -h, --help            show this help message and exit
  --format {markdown,latex}
```

---

<!-- page: validation -->

## Validation and fidelity

Validation has three parts:

1. Numerical correctness: does the code solve the stated equations accurately? (tested, below)
2. Model fidelity: are the equations a good description of a real aircraft? (depends on calibration)
3. Implementation parity: do the CPU kernel, NumPy and torch paths agree? (tested)

### What is tested (and where)

| Check | Reference | Tolerance | Test |
|---|---|---|---|
| Free fall `z = z₀ − ½ g t²` | closed form | 1e-8 m | `test_physics.py::test_free_fall_matches_analytic` |
| Hover is an exact equilibrium | `T = mg` | 1e-9 | `test_hover_is_equilibrium` |
| Terminal velocity (quadratic drag) | `√(mg/c)` | 1e-4 rel | `test_terminal_velocity_quadratic_drag` |
| Linear-drag velocity history | `v(t) = (g/k)(1−e^{−kt})` | 1e-9 rel | `test_linear_drag_analytic_solution` |
| Wind relaxation | `v = w(1−e^{−kt/m})` | 1e-8 rel | `test_wind_pulls_vehicle_toward_wind_speed` |
| Torque-free rotation | conservation of `L` and `T_rot` | 1e-8 rel | `test_torque_free_rotation_...` |
| Roll/yaw torque | `J⁻¹ A w²` | 1e-6 rel | `test_roll_and_yaw_torque_match_allocation` |
| Gyroscopic coupling | `ω̇_y = ω_x h_z / J_yy` | 1e-4 rel | `test_gyroscopic_coupling` |
| Motor lag | `w_cmd(1 − e^{−t/τ})` | 1e-9 rel | `test_motor_first_order_lag` |
| Ground effect | Cheeseman–Bennett | 1e-4 | `test_ground_effect_matches_cheeseman_bennett` |
| ISA density/pressure | standard tables | 0.1 % | `test_isa_atmosphere_reference_values` |
| Gravity models | WGS-84 / inverse-square | 1e-4 | `test_gravity_models` |
| Integrator order | Euler 1, midpoint 2, RK4 4 | ±0.35 | `test_integrator_convergence_order` |
| Stiff motor stability | auto sub-stepping | no blow-up | `test_stiff_motor_lag_is_stable_via_substeps` |
| Dryden statistics | `std(w) = σ_w`, `std(u) = σ_u` | 12 % | `test_dryden_turbulence_statistics` |
| Impact speed & rest | `√(2gh)` | 0.5 % | `test_ground_contact_and_impact_speed` |
| Energy accounting | `∫P dt` | 1e-9 rel | `test_battery_energy_and_soc_consistency` |
| Ray casting | independent AABB reference, analytic spheres/cylinders | 1e-9 | `test_sensors.py` |
| IMU noise | configured density | 8 % | `test_imu_noise_density_matches_config` |
| FSPL, LoS curves, noise floor | textbook values | 0.01 dB | `test_pipeline.py` |
| Fast Numba kernel vs reference | every integrator, vehicle, gravity/atmosphere variant, terrain, collisions, failures, battery | 1e-8 | `test_fast_parity.py` |
| Torch float64 vs NumPy | full simulation | 1e-9 | `test_torch.py` |
| Autograd through the dynamics | finite differences | 1e-5 rel | `test_autograd_through_dynamics_...` |
| Bit-exact reproducibility | re-simulate from archive | 0.0 | `test_reproducibility_is_bit_exact...` |
| ESKF strapdown kinematics | closed form of a banked turn | 2e-4 m/s (v), 2e-3 m (p) | `test_strapdown_propagation_matches_the_closed_form_of_a_banked_turn` |
| ESKF error transition | finite differences of the nominal propagation | Jacobian match | `test_error_transition_matches_finite_differences_...` |
| ESKF statistical consistency | χ² acceptance interval of average NEES / NIS | inside the 95 % band | `test_monte_carlo_nees_and_nis_are_consistent_...` |
| Delayed fusion | identical to on-time fusion | numerical equality | `test_delayed_measurements_give_the_same_estimate_...` |
| Occupancy ray traversal | dense sampling of the ray | same voxel set | `test_the_traversal_visits_exactly_the_voxels_...` |
| A\* | Dijkstra on random grids | equal cost | `test_astar_is_optimal_against_dijkstra_on_random_grids` |
| D\* Lite | a fresh search after every map change | equal cost | `test_dstar_lite_matches_a_fresh_search_after_every_map_change` |
| Projection and QP solver | generic solvers (SLSQP) on random problems | solver tolerance | `test_projection_matches_...`, `test_qp_matches_slsqp_...` |
| Minimum-snap | classic 7th-order rest-to-rest polynomial; lower snap than other interpolants | 1e-9 abs | `test_one_segment_rest_to_rest_is_...` |
| Time-optimal profile | bang-bang trapezoid on a straight line | profile match | `test_straight_line_profile_is_the_trapezoid_of_bang_bang_control` |
| Propeller scaling | thrust ∝ n², power ∝ n³, thrust ∝ ρ, CT/CP invariant under geometric scaling | 0.2 % (0.5 % for scaling) | `test_thrust_and_power_obey_the_propeller_scaling_laws` |
| Motor and ESC | closed-form operating point; power balance; ODE time constant | see test | `test_vehicle_realism.py` |
| Battery energy | chemistry = terminals + heat + stored change | 2 % | `test_energy_is_accounted_for_between_chemistry_heat_and_the_terminals` |
| Tethered load | pendulum period; static tension = weight; energy conservation | 1.5 % (period) | `test_a_load_on_a_fixed_attachment_swings_as_a_pendulum`, `test_static_load_tension_...` |
| Fixed-wing trim | equilibrium that holds for a minute; turn bank angle from load factor | see test | `test_straight_level_trim_is_an_equilibrium_...`, `test_a_coordinated_turn_trims_...` |
| Drag-free phugoid | Lanchester period (first-order theory); energy conserved | 12 % (period), 0.2 % (energy) | `test_the_phugoid_of_a_drag_free_glider_follows_lanchester` |
| MAVLink codec | published CRC values and CRC_EXTRA constants; round trip of every message | exact | `test_bridges_mavlink.py` |
| System identification | parameters of a perturbed simulated vehicle, noisy log | 1-4 % | `test_five_parameters_are_recovered_from_a_noisy_flight` |
| Consensus rate | second-largest `|1 − ε λ_i|` of the Laplacian | 2e-3 | `test_consensus_preserves_the_mean_and_decays_at_the_rate_the_spectrum_predicts` |
| Graph spectra | closed forms of rings, paths, complete graphs and stars | 1e-12 | `test_the_ring_and_path_spectra_match_their_closed_forms` |
| Hungarian | `scipy.optimize.linear_sum_assignment` (random, rectangular, forbidden pairs) | 1e-9 | `test_hungarian_matches_scipy` |
| Auction | exact optimum within `m ε` | the guaranteed bound | `test_auction_is_within_its_guaranteed_distance_of_the_optimum` |
| CBBA | brute-force optimum; conflict-free; sparse graphs | ≥ 50 % (guarantee), ≥ 85 % observed | `test_cbba_is_conflict_free_and_within_the_guarantee_of_the_optimum` |
| Lloyd coverage | uniform segment: sites `(2i−1)/(2n)`, cost `1/(12n²)` | 2e-3 | `test_lloyd_on_a_uniform_segment_finds_the_analytic_optimum` |
| 4-D conflicts | roots of the relative-distance quadratic | 1e-9 | `test_conflict_intervals_are_the_roots_of_the_quadratic` |
| Space-time planner | re-checked with the exact conflict detector | no conflicts | `test_two_vehicles_swap_places_in_a_corridor_wide_enough_to_pass` |
| Sobol indices | Ishigami and g-function analytic indices | 0.03 | `test_sobol_indices_of_the_ishigami_function_match_the_analytic_values` |
| Morris | linear model: `μ*` = coefficients, `σ` = 0 | 1e-12 | `test_morris_recovers_a_linear_model_exactly_and_flags_interactions` |
| Gaussian process | log marginal likelihood vs `scipy.stats.multivariate_normal`; textbook posterior | 1e-8 | `test_log_marginal_likelihood_and_posterior_equal_the_textbook_formulas` |
| Bayesian optimisation | Branin global minimum 0.3979 in 30 evaluations | median < 0.7, beats random | `test_bayesian_optimisation_finds_the_branin_minimum_...` |
| Stability margins | `2/(s(s+1)(s+2))`: GM = 3 at `√2`; `1/(s(s+1))` PM | 1e-6 | `test_stability_margins_match_the_textbook_values` |
| ROC and AUC | scikit-learn (ties included); Gaussian classes `Φ(d′/√2)` | 1e-12; 0.01 | `test_roc_curves_have_the_right_extremes_and_agree_with_scikit_learn` |
| Optical flow | plane expansion `Z/(Z−d)`, shift `f t/Z`, rotation homography `K Rᵀ K⁻¹` | 1e-9 | `test_flow_of_a_fronto_parallel_plane_matches_the_closed_forms` |
| Flow vs rendered frames | same segmentation id at the flowed pixel | ≥ 97 % of non-occluded pixels | `test_flow_from_rendered_frames_predicts_the_same_surface_in_the_next_frame` |
| Segmentation boxes | analytic projection of a fronto-parallel box | 1 px | `test_segmentation_boxes_match_the_analytic_projection_of_a_fronto_parallel_box` |
| PNG writer and decoder | Matplotlib's decoder; all five scan-line filters | exact | `test_png_round_trips_and_matplotlib_reads_it_the_same` |
| Policy export | ONNX Runtime vs the PyTorch policy | 1e-5 | `test_policies_save_load_and_export_to_onnx_and_torchscript_with_identical_outputs` |
| Resumable experiments | killed run resumed from the cache equals an uninterrupted run | 0.0 | `test_an_interrupted_run_resumes_from_the_cache_and_equals_an_uninterrupted_one` |

Run everything with `pytest`.

### Fidelity by subsystem

| Subsystem | Level | Notes |
|---|---|---|
| Rigid body, motors, gyroscopics | Verified numerically; physically standard | Quadratic thrust/torque in `w`; no blade-flapping or rotor–rotor interaction. |
| Drag, ground effect, inflow | Representative | Coefficients of presets are order-of-magnitude, not measured. Identify from flight logs for quantitative work. |
| Battery | Simple by default; detailed on request | The default is an equivalent circuit. The `battery_model` hook adds Peukert capacity, temperature, a lumped thermal model and ageing (generic parameters, not a measured cell). |
| Contact | Simple | Sphere-vs-primitive with restitution/friction; no rigid-body stacking or deformable terrain. |
| Atmosphere, gravity | Standard models | ISA troposphere only (< 11 km). |
| Wind | Standard models | Dryden is a stochastic spectral model, not CFD. `urban` adds an engineering cavity-wake model (Fackrell) behind buildings, not separation bubbles or vortex shedding; `gridded` imports CFD results. |
| IMU | Allan-variance model, verified against analytic curves | White, random-walk and 1/f terms (flicker is a four-process approximation, ±10 % plateau ripple), vibration aliasing, thermal drift, g-sensitivity. Grade presets are representative, not a specific part. |
| GNSS | Pseudorange model, geometry verified | WLS solution, DOP, NLoS bias, jamming. Constellations are synthetic Walker patterns (right statistics, not a date's real almanac); no carrier phase / RTK. |
| Baro/mag/pitot/flow/sonar/UWB | Typical, configurable | Magnetic field is a tilted dipole (global pattern, no regional anomalies); pitot has lag and dynamic-pressure effects; sonar has cone beam and speed-of-sound error. |
| LiDAR/depth | Geometric | Ray casting against primitives, terrain and BVH meshes with noise/dropout, scan patterns, motion distortion, Doppler and weather extinction; first surface only, no multipath, no material reflectivity beyond Lambertian incidence. |
| Radar | Radar equation + first surface | FMCW resolution and ambiguity verified analytically, coherent-integration SNR, CA-CFAR; no multipath, one target per beam, simple ground clutter. |
| Camera, event camera | Stylised ray-cast | Pinhole/fisheye lenses, rolling shutter, blur, exposure, sun and weather; not photorealistic, no textures; expect a visual domain gap for vision-based sim-to-real. |
| Environment | Geometric | WGS84 geodesy and NOAA sun position are accurate; weather changes density and visibility; imported meshes are static and used as given (no repair). |
| Wireless | Statistical + geometric LoS | Al-Hourani mean excess loss with optional ray-cast LoS; no ray-traced multipath or beamforming. |
| Estimator | ESKF, verified for consistency | Error-state Kalman filter with delayed fusion and χ² gating, NEES/NIS checked by Monte Carlo with white measurement noise. Against the simulator's slowly wandering GNSS error the reported covariance is optimistic (see [State estimation](#state-estimation)). The complementary filter and PV-KF remain as simpler options. |
| Propeller, motor, ESC | Blade-element model, generic data | BEMT with Prandtl losses; blade geometry and airfoil polars are synthetic and typical, not a measured propeller. Static figure of merit ≈ 0.58. |
| Slung load, proximity | Representative | Elastic-rope point-mass pendulum; ceiling and wall effects are first-order corrections. No rope aerodynamics or rotor-wake interaction. |
| Planning, avoidance | Algorithms verified, guarantees conditional | Optimality/equivalence tests for the searches; ORCA and CBF guarantees hold for their kinematic models and are measured in simulation, not proven for the rotor-level dynamics. |
| Advanced controllers | Verified in simulation | MPC, MPPI, INDI, L1 and sliding mode against the geometric controller with disturbances; stability margins are not certified. |
| Fixed-wing, VTOL | Reduced-order textbook models | Beard & McLain Aerosonde; linear coefficients with a post-stall blend; no slipstream or rotor–wing interference. Not validated against any real airframe. |
| MAVLink, ULog, HIL, ROS 2 | Implemented to the definitions; not run against real software | Codec checked against published constants, HIL against an in-process autopilot, ROS 2 against a stand-in for `rclpy`. |
| System identification | Recovers the simulator's own parameters | Multiple shooting; standard errors are optimistic; mass/`kT` and `kQ`/`Jz` are degenerate. Not demonstrated on real flight logs. |
| Multi-agent algorithms | Verified against closed forms and exact solvers | Consensus, allocation, coverage and 4-D deconfliction are checked as above. Their guarantees assume the idealised models (connected graph, reliable or symmetric links, point agents); the engine's link is lossy and delayed. |
| Swarm laws | Heuristic, tested in simulation | `formation_keeping` is a reference-shifting rule without a stability proof; the tests measure the formation error with and without the link. |
| Swarm environment | Cooperative bookkeeping in NumPy | Dynamics run on any backend; neighbour lists and the coverage cost are NumPy. A PettingZoo-style API, checked by hand (PettingZoo is not installed). |
| Additional tasks and experts | Scripted experts solve each task | The tests require 95 % (gates, landing), 90 % (inspection, coverage) and 70 % (target tracking, judged at the final instant, limited by the velocity loop's lag); the experts reach about 100 % in practice. Task rewards are design choices, not benchmarks from the literature. |
| Imitation, DAgger, export | Verified on the landing task | DAgger lifts a cloning policy from about 0.6 to 1.0 success on a few hundred aggregated states; this is one task, not a general claim. |
| Vision datasets | Exact labels, stylised images | Boxes, depth and flow are exact for the geometry the ray caster has; the pictures are not photorealistic, so there is a domain gap to real images. Moving obstacles break the rigid-scene flow. |
| Experiments toolkit | Verified on benchmark functions | Sobol and Morris on analytic functions, the GP against the textbook formulas, the cache against an interrupted run. The Slurm script is generated and checked as text, never submitted to a real cluster. |
| Analysis figures and templates | Numerics tested, widths from publisher guidelines | Margins, ROC and density numerics are tested; the column widths are the publishers' published values as of writing and should be checked against your template. |

### Calibrating to a real vehicle

1. Measure mass, inertia (bifilar pendulum), arm geometry, rotor spin directions.
2. Fit `kT, kQ, τ_m` from static thrust-stand data (`T` vs `w²`, step response).
3. Fit drag from steady-state tilt vs speed in a straight-line flight log.
4. Put the numbers in a `VehicleSpec`, replay your logged commands, and compare `pos/vel/attitude`.

Sim-to-real conclusions should always be accompanied by this comparison.

### Wireless

The wireless suite has its own validation table (analytic references, hand calculations, published figures) in [Wireless](#validation-summary).
Its fidelity limits are stated there: system-level abstractions, first-order specular ray tracing, an interval-level NR stack, and 3GPP/ITU constants
that are verified against the formulas but not against an independent implementation.

---

<!-- page: roadmap -->

## Roadmap to 1.0

Aevion 0.1.0 is the first release. It provides the simulation core, sensors, wireless, planning, estimation, control, multi-agent, learning and analysis
components listed in the [changelog](#changelog). The work remaining before 1.0 is an API freeze, a published validation report and a research-grade benchmark suite.

### Design rules

A module is complete when it has:

1. an abstract interface with a documented contract;
2. at least three implementations, from simple to high fidelity;
3. a registry entry, so configurations and third-party plugins can refer to it by name;
4. a typed, validated configuration schema;
5. tests, including at least one check against an analytic or published reference;
6. a documentation section and a runnable example;
7. a stated fidelity level in the [validation](#validation-and-fidelity) section.

Other principles: the core stays light (heavy dependencies are optional extras; the base install is NumPy and SciPy); existing tools are bridged rather
than rebuilt (Sionna for ray-traced radio, Isaac or Unreal for photorealistic rendering, PX4 SITL for autopilot fidelity).

### Known limitations

Validation and CI
- The GitHub workflows have not run on GitHub; they are checked structurally and through the equivalent local commands (`tools/ci_local.py`).
- The benchmark baseline comes from one machine, so the CI benchmark job reports only.
- The CUDA backend is untested (no NVIDIA GPU was available). The Python 3.10 leg is checked statically only.

Interoperability
- HIL, the MAVLink and ULog bridges and the ROS 2 adapter follow the public definitions and are tested against Aevion itself and stand-ins, not against
  PX4, ArduPilot or a ROS 2 installation.
- Sionna and ns-3 are not bridged; the exporters write files they can read.
- The TensorBoard, MLflow and W&B loggers, the PettingZoo-style API and the Slurm script are tested against stand-ins or as text.
- Not built: RLlib, TorchRL and skrl adapters, Ray and Dask runners, UKF and particle filters, CasADi/acados MPC, MuJoCo and PyBullet adapters.

Wireless
- Ray tracing is first-order specular (ground and box faces) with Bullington diffraction: no diffuse scattering or multi-bounce.
- The NR stack abstracts at the network-interval level (attenuated-Shannon BLER, no per-slot HARQ, no CSI feedback, full-buffer interference).
- The 3GPP and ITU-R constants were checked against hand calculations, not an independent implementation. Verify absolute values before quoting them.
- The UAV propagation default is `tr36777`; `tr38901` is flagged for UAV heights.

Environment and sensors
- The magnetic field is a tilted dipole (no regional anomalies); GNSS constellations are synthetic Walker patterns, not real almanacs.
- Radar and LiDAR return the first surface only. The camera is a ray-cast approximation; vision datasets have no photorealism.
- The urban wind field is an engineering wake model, not CFD. Meshes and moving obstacles are kinematic: obstacles do not react to the vehicle.
- GNSS spoofing is a measurement-level fault; the receiver has no detection logic.

Estimation, control and vehicles
- The ESKF covariance is optimistic against the simulator's slowly wandering GNSS error, because it treats fixes as white noise.
- Propeller geometry and airfoil data are generic. Fixed-wing and VTOL models are reduced-order.
- System identification recovers a simulated vehicle's parameters; its standard errors are optimistic, `kT` and mass (and `kQ` and `Jz`) are degenerate from
  flight data alone, and it has not been used on real logs.
- ORCA and CBF guarantees hold for their kinematic models, not for rotor-level dynamics. MPPI needs a few hundred samples; sliding mode needs a gain larger
  than the disturbance.

Multi-agent, learning and viewer
- Multi-agent guarantees assume idealised links; under the engine's lossy link the swarm laws are tested, not proven.
- Task rewards are design choices, not literature benchmarks. DAgger was verified on the landing task only.
- The viewer's 1 h / 50-vehicle target of 30 fps on integrated graphics has not been measured. Not implemented: RGB and depth panels, wind streamlines,
  HDRI sky and satellite-tile terrain.
- Journal-template column widths are the publishers' guideline values; check them against the template you submit to.

### Planned

- API freeze, validation report and benchmark suite for 1.0.
- Validation of the bridges against PX4, ArduPilot and ROS 2; of the wireless constants against an independent implementation.
- A measured viewer frame-rate benchmark, CUDA testing, a real CI run and a runner-class benchmark baseline.
- Optional extras: IGRF geomagnetics, real GNSS almanacs, UKF, nonlinear MPC and RLlib/TorchRL adapters.

---

<!-- page: changelog -->

## Changelog

### 0.1.0 (initial release)

#### Added
- Core: typed registries with plugin discovery; a versioned, validated configuration schema (YAML/JSON, JSON Schema export, content hashing); the
  `Simulation` engine (multi-rate scheduler, event bus, hooks, named RNG streams, fault injection); health monitors; frame and clock models; `reference`
  and `numba` dynamics kernels with a parity guard; chunked, streaming, hash-verified archives with provenance, a run registry and `aevion diff`.
- Dynamics and vehicles: Newton–Euler multirotors with RK4 and other integrators; blade-element-momentum propellers, BLDC and ESC models, battery packs,
  slung loads, ground, ceiling and wall effects; fixed-wing, quad-plane and tilt-rotor aircraft with trim, linearisation and autopilots.
- Environment: WGS84 geodesy, geomagnetic model, solar ephemeris, weather; log-law, gridded and urban-wake wind; triangle meshes with a BVH (OBJ, STL,
  glTF); terrain and OpenStreetMap/GeoJSON loaders; signed-distance fields; moving obstacles; geofences.
- Sensors: Allan-variance IMU, pseudorange GNSS, barometer, magnetometer, pitot, optical flow, ultrasonic, UWB, FMCW radar, LiDAR scan patterns,
  camera models (lenses, rolling shutter, thermal), event camera, and a sensor-fault library.
- Estimation, planning and control: 15-state ESKF with delayed fusion and NEES/NIS tools; occupancy mapping; A\*, Theta\*, D\* Lite, RRT\*, PRM;
  minimum-snap and time-optimal trajectories; ORCA-3D and CBF avoidance; behaviour trees; geometric, PID and LQR controllers plus MPC, MPPI, INDI, L1 and
  sliding-mode control.
- Wireless: 3GPP TR 38.901 and TR 36.777 propagation, ray tracing, fading, UAV antennas and beamforming, MIMO, an NR-style cellular network (handover,
  RLF, schedulers, link adaptation), RIS, satellite links, radio maps, jamming and spoofing, link metrics and ns-3 export.
- Multi-agent: communication graphs, consensus, Hungarian, auction and CBBA allocation, Voronoi coverage, 4-D conflict detection and space-time planning,
  and swarm laws applied through the lossy link.
- Machine learning: Gymnasium and tensor-native vectorised environments, swarm environments, gate-racing, landing, tracking, inspection and coverage
  tasks with scripted experts, curricula, behaviour cloning and DAgger, ONNX and TorchScript export, tracking loggers, Optuna, vision datasets (COCO, YOLO,
  depth, optical flow) and differentiable dynamics.
- Experiments and analysis: designs of experiments, Morris and Sobol sensitivity, Bayesian optimisation, resumable cached runs, Slurm job arrays, a
  SQLite result database, benchmark suites and leaderboards; IEEE, ACM, Elsevier and Springer figure templates, control and estimation diagnostics, and an
  auto-generated methods appendix.
- Bridges: MAVLink v1/v2, `.tlog` and PX4 ULog files, HIL over MAVLink, a ROS 2 adapter and system identification.
- Viewer: a dependency-free web viewer with single-file export, a local server, live streaming, run comparison, video and screenshot export, a notebook
  widget and a scene editor.
- Tooling: the `aevion` command line, a plugin scaffolder, untrusted-configuration limits (`--untrusted`), property-based and golden-image tests,
  benchmarks with a regression check, a public-API snapshot, a deprecation framework, a local CI runner and GitHub Actions workflows.

#### Known limitations
See [Known limitations](#known-limitations) in the roadmap.

---

<!-- page: contributing -->

## Contributing to Aevion

This guide covers setting up a checkout and getting a change merged.

### Set up

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                   # tests, lint, types, docs tooling
pre-commit install                                        # optional: run the fast checks on every commit
```

Optional extras for specific areas: `.[fast]` (Numba kernel), `.[torch]`, `.[gym]`, `.[viz]`, `.[all]`.
Python 3.10-3.13 are supported; do not use syntax or library features newer than 3.10.

### The checks every change must pass

```bash
python tools/ci_local.py            # runs everything below the way CI does
```

or individually:

| Gate | Command |
| --- | --- |
| Lint and format | `ruff check . && ruff format --check .` |
| Types | `mypy src/aevion` |
| Tests + coverage (>= 85 %) | `pytest -n auto --cov=aevion` |
| Public API unchanged (or deliberately updated) | `python tools/api_snapshot.py` |
| Generated docs in sync | `python tools/gen_config_docs.py --check` |
| Performance | `python tools/bench_check.py` |
| Security | `bandit -r src -c pyproject.toml` |
| Docs build | `mkdocs build --strict` |

### Where things live

The [Architecture](#architecture) section explains the layers. Short version: `config/` (typed schema) -> `registry/` (components by
name) -> `engine/` (the `Simulation` loop) -> `datasets/` (archives). Physics is in `dynamics/` and `core/`, sensing in
`sensors/`, networking in `wireless/`, figures in `analysis/`, the 3-D viewer in `viewer/`.

### Writing code

- Match the surrounding style; `ruff format` decides layout. Line length is 150.
- Public functions and classes get type annotations and a docstring that states units and frames
  (world ENU, body FLU, quaternion `[w,x,y,z]`, SI units unless the name says otherwise).
- Array code must work for the NumPy and torch backends: go through the `Backend` object (`B.sin`, `B.stack`, ...)
  rather than calling `np.` inside code that torch also runs.
- New components are registered (`registries.sensors.register(...)`) and documented in `docs/`; do not add
  `if name == ...` chains.
- No `pickle`, `eval`, `exec`, `shell=True`, unsafe YAML loading or network access (the security test fails on them).
- Never import optional heavy dependencies (torch, numba, pyarrow) at module import time; import them where used, and
  read versions through `importlib.metadata`.

### Tests

- Every bug fix comes with a test that fails before the fix.
- Physics changes need a validation test against an analytic or published reference (see [Validation and fidelity](#validation-and-fidelity)).
- Prefer properties (`tests/test_properties.py`, hypothesis) for invariants and parsers.
- Results that must not drift live in `tests/golden/`. If a change is *supposed* to alter them, regenerate with
  `python tools/update_goldens.py` (figures) or the script named in the test, review the diff, and explain it in
  the CHANGELOG ("Changed (results)").
- Mark slow tests `@pytest.mark.slow`, GPU tests `@pytest.mark.gpu`; they are skipped when the hardware is missing.

### Changing public API

Read the [versioning policy](#versioning-and-deprecation-policy). In short: add the new thing, deprecate the old one with
`aevion.deprecation`, keep it for two minor releases, and update `tests/golden/public_api.json` with
`python tools/api_snapshot.py --update` in the same commit as a changelog entry.

### Commits and pull requests

- One logical change per PR; describe *why*, not only what. Link the issue.
- Update the [changelog](#changelog) (Unreleased section) for anything user-visible.
- Documentation changes are code changes: executable examples in `DOCUMENTATION.md` are tested.
- Be kind in review. See the [code of conduct](#code-of-conduct).

### Adding a plugin or component

`aevion new-plugin my_package` scaffolds a plugin package (entry point, example sensor, tests). The mechanism is
described in [Writing a plugin](#writing-a-plugin). Components that are broadly useful can be proposed for the core package via an issue.

---

<!-- page: code_of_conduct -->

## Code of conduct

Aevion is a research tool built by people with different backgrounds, languages and levels of experience. We want
participation in the project to be productive and respectful for everyone.

### Our standards

Expected:

- Assume good faith; ask questions before assigning blame.
- Be specific and constructive in review: criticise code and ideas, not people.
- Credit others' work and cite sources, models and data you build on.
- Respect people's time: provide minimal reproducing examples, search before opening duplicates.
- Accept that maintainers must sometimes say no to protect the project's scope and quality.

Not acceptable:

- Harassment, insults, discriminatory jokes or slurs, personal or political attacks.
- Sexualised language or attention, intimidation, stalking, or publishing others' private information.
- Sustained disruption of discussions, or pressure to merge or answer on someone's schedule.
- Using the project to develop or promote systems intended to cause harm to people.

### Scope

This applies in all project spaces (issues, pull requests, discussions, chats, events) and when someone represents the
project in public.

### Enforcement

Maintainers may edit or remove comments, lock threads, and warn, temporarily restrict or permanently ban people who
violate these standards. Report concerns privately to the maintainers (see the [security policy](#security-policy) for the
contact route); reports are handled confidentially and the reporter will not face retaliation. Maintainers who are
involved in an incident recuse themselves from deciding it.

This code of conduct is adapted from widely used open-source community covenants.

---

<!-- page: security -->

## Security policy

### Reporting a vulnerability

Please do not open a public issue for security problems. Report them privately through the repository's
"Report a vulnerability" (private security advisory) feature, or e-mail the maintainers at the address listed on the
project page. Include the Aevion version (`aevion info`), a minimal reproducing configuration or archive, and the
impact you expect. You will get an acknowledgement within 5 working days and a fix or mitigation plan within 30 days
for confirmed issues. Credit is given in the release notes unless you prefer otherwise.

### Supported versions

Security fixes are released for the latest minor version. While Aevion is `0.x`, older minors are not patched.

### Threat model

Aevion is a numerical research library. It runs on your machine with your privileges. What it protects against, and
what it does not:

Treated as untrusted input, and hardened

| Input | Protections |
| --- | --- |
| Configuration files (YAML/JSON) | `yaml.safe_load` only; size, nesting-depth, node-count and YAML-alias limits; unknown fields are reported, never executed. `aevion run --untrusted` additionally enforces resource limits (duration, steps, vehicles, sensor ray rates, archive size, wall-clock time), confines output to one directory and allows built-in components only (`aevion.safety`). |
| Run archives (`.aevd`, `.aevz`, `.aev`) | Manifest schema validation (numeric dtypes only, bounded shapes and sizes); chunk paths must be safe relative paths; arrays are loaded with `allow_pickle=False`; `.aevz` extraction rejects absolute paths, `..`, drive letters, symlinks and zip bombs (size and compression-ratio caps), and removes partial output on failure. BLAKE2b hashes detect corruption (`aevion verify`). |
| The viewer export | Data is embedded as numeric arrays and JSON; no archive content is interpreted as code. The only network access is the opt-in download of pinned three.js files for `--offline`, HTTPS only. |

Trusted code (not sandboxed)

- Plugins (`aevion.plugins` entry points) and custom hooks/components are ordinary Python code running with your
  privileges. Install only plugins you trust. `AEVION_DISABLE_PLUGINS=1` skips plugin loading entirely.
- Python objects you construct yourself (callables passed as hooks, controllers, health checks) are trusted.

Not provided

- Resource limits bound *accidental or naive* abuse; they are not a sandbox against arbitrary code. To run
  third-party code, use OS-level isolation (containers, VMs, separate users).
- Aevion makes no claims about the physical safety of controllers developed with it; it is a simulator, not flight
  software.

### Engineering measures

- No `pickle`, `eval`, `exec`, `os.system`, `shell=True`, unsafe YAML loading or cleartext URLs in the package; a
  static test (`tests/test_security_static.py`) enforces this on every commit.
- `bandit` runs in CI with no tolerated medium/high findings; `pip-audit` checks dependencies.
- Property-based fuzzing (`tests/test_properties.py`) feeds arbitrary documents to the configuration loader; the
  only permitted outcomes are a valid configuration or a `ConfigError` that names the offending field.
- Releases are built in CI from tagged commits and smoke-tested in a clean environment before upload.
- Provenance stored in archives contains no personal data (no user name, host name or absolute paths).

---

<!-- page: versioning -->

## Versioning and deprecation policy

Aevion follows [Semantic Versioning 2.0](https://semver.org). While the major version is `0`, the rules below apply
with one relaxation: a minor release (`0.x.0`) may contain breaking changes, but only after the deprecation period
described here. From `1.0.0` on, breaking changes happen only in major releases.

### What is public API

Everything importable from the modules listed in `src/aevion/_api.py` (`PUBLIC_MODULES`) under a name that does not
start with an underscore, plus:

- the command-line interface (`aevion ...` sub-commands, flags and exit codes);
- the configuration schema (`schema_version`, field names, their meaning and units);
- the archive format (`aevion.run/N` manifests and the channel names/units documented in [Run archives](#run-archives));
- the plugin entry-point group `aevion.plugins` and the registry interfaces in [Writing a plugin](#writing-a-plugin).

Anything else (names with a leading underscore, `aevion.<pkg>._internal`, test helpers, `tools/`, `benchmarks/`) is
private and may change without notice.

`tests/test_public_api.py` enforces this: it compares the live API with `tests/golden/public_api.json`. A removed
name, a removed or reordered parameter, or a new required parameter fails the test. New public names also fail until
the snapshot is updated deliberately (`python tools/api_snapshot.py --update`) together with a CHANGELOG entry.

### What counts as breaking

| Change | Breaking? |
| --- | --- |
| Removing or renaming a public name, parameter, CLI flag, config field | yes |
| Adding a required parameter / config field without a default | yes |
| Changing a default so that results of existing configs change | yes |
| Changing simulated physics for an existing model (different trajectory for the same config) | yes, unless it fixes a documented error (see below) |
| Changing numerical results by floating-point round-off only | no |
| Adding optional parameters, config fields, registry entries, CLI flags, archive channels | no |
| Tightening validation to reject configs that were previously *silently misinterpreted* | no (bug fix) |
| Dropping support for a Python version | no, but announced one minor release in advance |

Physics corrections. A bug fix that changes simulated results is allowed in a minor release, must be listed under
"Changed (results)" in the CHANGELOG with the affected models, and must update the golden data
(`tests/golden/`) in the same commit so the change is visible in review. Archives record the Aevion version, so old
results stay attributable.

### Deprecation cycle

1. Add the replacement first; keep the old name working.
2. Mark the old one with `aevion.deprecation.deprecated(...)`, `renamed_kwarg(...)` or `warn_deprecated(...)`, giving
   `since`, `remove_in` and the `alternative`. The warning class is `AevionDeprecationWarning` (a `FutureWarning`, so
   end users see it by default).
3. `remove_in` must be at least two minor releases after `since`; `deprecation.register` rejects shorter periods.
4. List it under "Deprecated" in the CHANGELOG.
5. Remove it in the announced release. `tests/test_deprecation.py` fails if the package version reaches a
   deprecation's `remove_in` while the feature is still present, so promises cannot silently lapse.
6. The library must not use its own deprecated API (a test runs the main workflows with the warning turned into an
   error).

### Schema and format versions

- Configuration: `schema_version` is an integer. Every older version has a registered migration in
  `aevion/config/migrate.py`, so old files keep loading; `aevion validate` reports what it migrated. A new schema
  version is only introduced when a field changes meaning or shape.
- Archives: `aevion.run/N`. A reader refuses archives newer than it understands (with an upgrade hint) and keeps
  reading all older ones. Writers never silently change the meaning of an existing channel.

### Release cadence

Patch releases (bug fixes, docs, performance) whenever needed; minor releases when features are ready. Each release
follows the [release process](#releasing-aevion).

---

<!-- page: releasing -->

## Releasing Aevion

Releases are cut from `main` by tagging. CI builds, checks and publishes; nothing is uploaded from a laptop.

### Checklist

1. Decide the version using the [versioning policy](#versioning-and-deprecation-policy). Search for deprecations that reach their `remove_in`
   version (`python -c "import aevion.deprecation as d; print(d.list_deprecations())"`); the test suite fails if any
   are overdue.
2. Update `version` in `pyproject.toml` and `src/aevion/__init__.py` (a test asserts they agree), and move the
   changelog "Unreleased" entries under the new version and date.
3. Run everything locally: `python tools/ci_local.py --full` (lint, types, tests with coverage, API snapshot,
   docs build, benchmarks, security, wheel build and clean-environment smoke test).
4. Refresh the baselines only if the change is intended: `python tools/bench.py --update-baseline`,
   `python tools/update_goldens.py`, `python tools/api_snapshot.py --update`. Review each diff.
5. Open a PR, wait for the full CI matrix (Linux/Windows/macOS x Python 3.10-3.13) to pass, merge.
6. Tag: `git tag -s vX.Y.Z -m "Aevion X.Y.Z" && git push origin vX.Y.Z`.
7. The `release` workflow builds the sdist and wheel, runs `twine check`, installs the wheel into a fresh virtual
   environment and runs `tools/smoke_wheel.py`, then publishes to PyPI with trusted publishing (no long-lived token).
8. Verify `pip install aevion==X.Y.Z` in a clean environment, run `aevion demo`, and publish the GitHub release
   notes (copy from the CHANGELOG).

### Hotfixes

Branch from the tag, cherry-pick the fix, bump the patch version, and follow the same steps. Never move or reuse a
tag.

### Yanking

If a release is broken, yank it on PyPI (do not delete), release a fixed patch version, and add a note under the
broken version in the CHANGELOG.
