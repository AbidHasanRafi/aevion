"""The extended sensor-fault library."""

from types import SimpleNamespace

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.engine.faults import FaultError
from aevion.engine.sensor_faults import SensorDelaySpikes, SensorDrift, SensorNaN, SensorSaturation, SensorScaleError, SensorSpikes
from aevion.sensors.base import Sensor

B = NumpyBackend()


class Dummy(Sensor):
    name = "imu"

    def measure(self, state):
        return dict(gyro=np.ones((3, 3)), accel=np.zeros((3, 3)), t=state.t)


def setup(fault, n=3, vehicle=None, t_start=0.0, t=0.0):
    s = Dummy(rate=100.0)
    sim = SimpleNamespace(dyn=SimpleNamespace(n=n), suite={"imu": s}, B=B, t=t)
    fault.vehicle, fault.t_start = vehicle, t_start
    fault.on_start(sim)
    return s, sim


def reading(**kw):
    base = dict(gyro=np.arange(9.0).reshape(3, 3) + 1.0, accel=np.zeros((3, 3)), t=0.0)
    base.update(kw)
    return base


def apply(s, r):
    for f in s.filters:
        r = f(r)
    return r


# ------------------------------------------------------------------------------------------- field faults
def test_scale_error_multiplies_only_the_targeted_vehicle_and_field():
    s, _ = setup(SensorScaleError("imu", "gyro", 2.0), vehicle=1)
    r = apply(s, reading())
    np.testing.assert_allclose(r["gyro"][1], 2.0 * np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(r["gyro"][[0, 2]], reading()["gyro"][[0, 2]])  # other vehicles untouched
    np.testing.assert_array_equal(r["accel"], 0.0)


def test_faults_end_cleanly_and_do_not_mutate_the_input():
    fault = SensorScaleError("imu", "gyro", 3.0)
    s, sim = setup(fault)
    original = reading()
    out = apply(s, original)
    np.testing.assert_allclose(original["gyro"], np.arange(9.0).reshape(3, 3) + 1.0)  # the sensor's own array is unchanged
    assert out["gyro"] is not original["gyro"]
    fault.on_end(sim)
    assert s.filters == []


def test_saturation_clips_to_the_limit():
    s, _ = setup(SensorSaturation("imu", "gyro", 4.5))
    r = apply(s, reading())
    assert r["gyro"].max() == 4.5 and r["gyro"].min() == 1.0 and r["gyro"][0, 0] == 1.0
    np.testing.assert_allclose(apply(setup(SensorSaturation("imu", "gyro", 4.5))[0], reading(gyro=-np.full((3, 3), 9.0)))["gyro"], -4.5)
    with pytest.raises(ValueError):
        SensorSaturation(limit=0.0)


def test_spikes_hit_a_fraction_of_samples_with_the_requested_magnitude():
    s, _ = setup(SensorSpikes("imu", "gyro", probability=0.1, magnitude=5.0, seed=3), n=3)
    hits, vals = 0, []
    for _ in range(3000):
        r = apply(s, reading(gyro=np.zeros((3, 3))))
        corrupted = np.any(r["gyro"] != 0.0, axis=1)
        hits += int(corrupted.sum())
        vals.extend(r["gyro"][corrupted].ravel())
    assert hits / 9000 == pytest.approx(0.1, rel=0.1)
    assert np.std(vals) == pytest.approx(5.0, rel=0.1)
    sign, _ = setup(SensorSpikes("imu", "gyro", probability=1.0, magnitude=2.0, mode="sign", seed=0))
    assert set(np.unique(apply(sign, reading(gyro=np.zeros((3, 3))))["gyro"])) == {-2.0, 2.0}


def test_spikes_are_reproducible_and_seed_dependent():
    def run(seed):
        s, _ = setup(SensorSpikes("imu", "gyro", 0.5, 1.0, seed=seed))
        return np.concatenate([apply(s, reading(gyro=np.zeros((3, 3))))["gyro"].ravel() for _ in range(20)])

    np.testing.assert_array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_drift_grows_linearly_from_the_start_of_the_fault():
    fault = SensorDrift("imu", "gyro", rate=[0.01, 0.0, -0.02])
    s, sim = setup(fault, t_start=2.0)
    for t, k in ((2.0, 0.0), (3.0, 1.0), (7.0, 5.0)):
        sim.t = t
        r = apply(s, reading(gyro=np.zeros((3, 3))))
        np.testing.assert_allclose(r["gyro"][0], k * np.array([0.01, 0.0, -0.02]), atol=1e-12)
    sim.t = 1.0  # before the start time: no negative drift
    np.testing.assert_allclose(apply(s, reading(gyro=np.zeros((3, 3))))["gyro"], 0.0)


def test_nan_fault_probability():
    s, _ = setup(SensorNaN("imu", "gyro", probability=0.2, seed=5))
    bad = sum(int(np.isnan(apply(s, reading())["gyro"]).any(axis=1).sum()) for _ in range(2000))
    assert bad / 6000 == pytest.approx(0.2, rel=0.1)
    always, _ = setup(SensorNaN("imu", "gyro", probability=1.0))
    assert np.isnan(apply(always, reading())["gyro"]).all()


def test_unknown_fields_and_sensors_are_reported_clearly():
    s, _ = setup(SensorScaleError("imu", "gyroscope", 2.0))
    with pytest.raises(FaultError, match="no field 'gyroscope'"):
        apply(s, reading())
    fault = SensorScaleError("nope", "gyro", 2.0)
    from aevion.sensors.base import SensorSuite

    sim = SimpleNamespace(dyn=SimpleNamespace(n=1), suite=SensorSuite(Dummy()), B=B, t=0.0)
    with pytest.raises(FaultError, match="no sensor named 'nope'"):
        fault.on_start(sim)
    for bad in (
        lambda: SensorSpikes(probability=2.0),
        lambda: SensorSpikes(mode="laplace"),
        lambda: SensorNaN(probability=-1),
        lambda: SensorDelaySpikes(delay_s=-1),
    ):
        with pytest.raises(ValueError):
            bad()
    assert SensorScaleError("imu", "gyro", 2.0).details()["factor"] == 2.0


# ----------------------------------------------------------------------------------------- latency spikes
def test_latency_spikes_delay_a_fraction_of_samples_and_preserve_order():
    s = Dummy(rate=100.0, latency=0.01)
    sim = SimpleNamespace(dyn=SimpleNamespace(n=1), suite={"imu": s}, B=B, t=0.0, n=3, world=None)
    s.bind(SimpleNamespace(B=B, n=3, world=None), 1)
    fault = SensorDelaySpikes("imu", probability=0.2, delay_s=0.35, seed=2)
    fault.on_start(sim)
    stamps, arrivals = [], []
    for k in range(3000):
        t = k * 0.01
        st = SimpleNamespace(t=t)
        if s.update(st):
            stamps.append(s.stamp)
            arrivals.append(t)
    lat = np.array(arrivals) - np.array(stamps)
    assert np.all(np.diff(stamps) > 0)  # delivery stays in order: late samples hold back later ones
    assert lat.min() == pytest.approx(0.01, abs=1e-9) and lat.max() >= 0.34
    assert np.mean(lat > 0.1) > 0.1  # a visible share of late deliveries (including the ones queued behind a late sample)
    fault.on_end(sim)
    assert s.latency_extra is None


# --------------------------------------------------------------------------------------------- engine
def test_faults_degrade_a_closed_loop_estimate_and_a_nan_does_not_poison_it():
    from aevion.engine import Simulation

    def run(faults):
        cfg = {
            "duration": 10.0,
            "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 3.0, "omega": 0.6}},
            "sensors": ["imu", "gps", "baro", "mag"],
            "estimator": {"type": "mahony_kf", "handover_s": 3.0},
            "faults": faults,
        }
        sim = Simulation(cfg)
        arch = sim.run()
        return arch, sim

    clean, _ = run([])
    nan, sim_nan = run([{"type": "sensor_nan", "t_start": 4.0, "params": {"sensor": "gps", "field": "pos", "probability": 0.3}}])
    spikes, _ = run([{"type": "sensor_spikes", "t_start": 4.0, "params": {"sensor": "gps", "field": "pos", "probability": 0.2, "magnitude": 30.0}}])
    for a in (clean, nan, spikes):
        assert a["est_pos"].shape == a["pos"].shape and np.all(np.isfinite(a["est_pos"]))  # NaN readings were rejected, not absorbed

    def est_err(a):
        sel = a.t - a.t[0] > 4.5
        return np.sqrt(np.mean(np.sum((a["est_pos"][sel] - a["pos"][sel]) ** 2, axis=-1)))

    assert est_err(spikes) > 1.5 * est_err(clean)  # outliers make the estimate worse
    assert any(e["type"] == "sensor_nan" for e in nan.meta["events"]) or any("fault" in e["name"] for e in nan.meta["events"])
    from aevion.registry import registries

    assert {"sensor_scale", "sensor_saturation", "sensor_spikes", "sensor_drift", "sensor_nan", "sensor_delay_spikes"} <= set(
        registries.faults.names()
    )
