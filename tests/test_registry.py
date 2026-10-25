import warnings

import pytest

from aevion.errors import RegistryError
from aevion.registry import Registry, load_plugin_function, registries


def test_register_create_and_aliases():
    r = Registry("things")

    @r.register("alpha", aliases=("a", "Alpha-One"))
    def make_alpha(x=1):
        """Make an alpha"""
        return ("alpha", x)

    assert r.create("alpha", x=3) == ("alpha", 3)
    assert r.create("A") == ("alpha", 1) and r.create("alpha_one") == ("alpha", 1)
    assert "a" in r and "zzz" not in r and r.names() == ["alpha"]
    assert r.entry("a").doc == "Make an alpha"


def test_duplicate_and_replace():
    r = Registry("things")
    r.register("x", lambda: 1)
    with pytest.raises(RegistryError, match="already registered"):
        r.register("x", lambda: 2)
    r.register("x", lambda: 2, replace=True)
    assert r.create("x") == 2
    with pytest.raises(RegistryError, match="collides"):
        r.register("y", lambda: 3, aliases=("x",))


def test_unknown_name_suggests_close_matches():
    with pytest.raises(RegistryError) as e:
        registries.controllers.create("geometrik", None)
    assert "Did you mean 'geometric'" in str(e.value) and "Available" in str(e.value)


def test_build_from_spec_and_defaults():
    r = Registry("things")
    r.register("p", lambda a=0, b=0: (a, b))
    assert r.build("p") == (0, 0)
    assert r.build({"type": "p", "a": 5}) == (5, 0)
    assert r.build({"kind": "p", "a": 5}, b=9) == (5, 9)  # kwargs act as defaults, spec wins
    with pytest.raises(RegistryError, match="no 'type'"):
        r.build({"a": 1})
    with pytest.raises(RegistryError, match="cannot create"):
        r.build({"type": "p", "nope": 1})


def test_all_builtin_components_are_registered():
    s = registries.summary()
    assert {"geometric", "cascaded", "lqr"} <= set(s["controllers"])
    assert {"imu", "gps", "baro", "mag", "rangefinder", "lidar", "camera"} <= set(s["sensors"])
    assert {"none", "constant", "shear", "gust", "dryden", "composite", "random_constant"} <= set(s["winds"])
    assert {"empty", "forest", "city", "corridor", "hills"} <= set(s["worlds"])
    assert {"rotor_failure", "gps_outage", "sensor_bias_step", "wind_gust", "comm_blackout"} <= set(s["faults"])
    assert {"ring", "line", "grid", "vee"} <= set(s["formations"])
    assert "crazyflie" in s["vehicles"] and "hover" in s["tasks"] and "finite" in s["health_checks"]
    assert registries.describe().count("\n") > 50


def test_v01_wind_class_names_still_resolve():
    w = registries.winds.build({"type": "DrydenTurbulence", "w20": 3.0, "severity": 1.0})
    assert type(w).__name__ == "DrydenTurbulence"
    c = registries.winds.build({"type": "CompositeWind", "models": [{"type": "ConstantWind", "velocity": [1, 0, 0]}, {"type": "Gust"}]})
    assert len(c.models) == 2


def test_plugin_registers_a_new_sensor_usable_from_a_config():
    from aevion.config import SimulationConfig
    from aevion.sensors.base import Sensor

    class Sonar(Sensor):
        name = "sonar"

        def __init__(self, rate=20.0, max_range=5.0, **kw):
            super().__init__(rate=rate, **kw)
            self.max_range = max_range

        def measure(self, state):
            return dict(range=state.pos[:, 2] * 0 + self.max_range, t=state.t)

    def register(regs):
        regs.sensors.register("sonar", Sonar, source="test_plugin")

    rec = load_plugin_function("test_plugin", register, registries)
    try:
        assert rec.loaded and "sonar" in registries.sensors
        assert registries.sensors.entry("sonar").source == "test_plugin"
        cfg = SimulationConfig.from_dict({"sensors": [{"type": "sonar", "max_range": 9.0}], "duration": 1.0})
        assert cfg.sensors[0].params == {"max_range": 9.0}
        from aevion.engine import Simulation

        sim = Simulation(cfg)
        sim.run()
        assert sim.suite["sonar"].latest["range"][0] == 9.0
    finally:
        registries.sensors.unregister("sonar")


def test_failing_plugin_is_isolated():
    def bad(regs):
        raise RuntimeError("boom")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rec = load_plugin_function("bad_plugin", bad, registries)
    assert not rec.loaded and "boom" in rec.error
    assert any("bad_plugin" in str(x.message) for x in w)
    assert len(registries.controllers) >= 3  # everything else intact


def test_introspection_of_accepted_parameters():
    from aevion.registry.introspect import accepted_params
    from aevion.sensors import IMU, Lidar

    imu = accepted_params(IMU)
    assert imu is not None and {"rate", "accel_noise_density", "latency", "dropout", "clock", "extrinsics"} <= imu
    assert "bogus" not in imu
    assert accepted_params(lambda a, **kw: 0) is None
    assert accepted_params(lambda sim, a, b=1: 0, skip_positional=1) == {"a", "b"}
    assert "channels" in accepted_params(Lidar)


def test_entry_point_discovery_with_fake_distributions(monkeypatch):
    """`discover` loads every entry point, isolates a broken one, and is idempotent."""
    from aevion.registry import plugins
    from aevion.sensors.base import Sensor

    class Dummy(Sensor):
        name = "dummy_ep"

        def measure(self, state):
            return dict(t=state.t)

    class EP:
        def __init__(self, name, target, fn=None, err=None):
            self.name, self.value, self._fn, self._err = name, target, fn, err
            self.dist = type("D", (), {"version": "9.9"})()

        def load(self):
            if self._err:
                raise self._err
            return self._fn

    good = EP("good_ep", "pkg:register", lambda regs: regs.sensors.register("dummy_ep", Dummy, source="good_ep"))
    broken_import = EP("broken_import", "pkg:missing", err=ImportError("no module named pkg"))
    broken_register = EP("broken_register", "pkg:register", lambda regs: 1 / 0)
    monkeypatch.setattr(plugins, "_entry_points", lambda: [good, broken_import, broken_register])
    monkeypatch.setattr(plugins, "_records", [])
    monkeypatch.setattr(plugins, "_loaded", False)
    try:
        with pytest.warns(UserWarning) as w:
            recs = plugins.discover(registries, force=True)
        by = {r.name: r for r in recs}
        assert by["good_ep"].loaded and by["good_ep"].version == "9.9" and "dummy_ep" in registries.sensors
        assert not by["broken_import"].loaded and "no module named pkg" in by["broken_import"].error
        assert not by["broken_register"].loaded and "ZeroDivisionError" in by["broken_register"].error
        assert len(w) == 2
        assert len(plugins.discover(registries)) == 3  # second call does not reload
    finally:
        if "dummy_ep" in registries.sensors:
            registries.sensors.unregister("dummy_ep")
