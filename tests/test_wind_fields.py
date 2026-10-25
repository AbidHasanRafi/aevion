"""Log-law, gridded and urban wind."""

import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.environments.wind import CompositeWind, ConstantWind, wind_from_dict
from aevion.environments.wind_fields import GriddedWind, LogLawWind, UrbanWind, fackrell_reattachment_length
from aevion.environments.world import World

B = NumpyBackend()


def sample(w, pts, t=0.0):
    w.reset(B, len(pts), 0)
    return np.asarray(w.sample(B, np.asarray(pts, float), np.zeros((len(pts), 3)), t, 0.01))


# ------------------------------------------------------------------------------------------------ log law
def test_log_law_matches_the_reference_speed_and_the_analytic_height_ratio():
    w = LogLawWind(speed_ref=8.0, height_ref=10.0, z0=0.1)
    assert sample(w, [[0, 0, 10.0]])[0, 0] == pytest.approx(8.0)
    z = 50.0
    assert sample(w, [[0, 0, z]])[0, 0] == pytest.approx(8.0 * math.log(z / 0.1) / math.log(10.0 / 0.1), rel=1e-12)
    assert w.u_star == pytest.approx(0.41 * 8.0 / math.log(100.0))


def test_log_law_displacement_direction_and_monotonicity():
    w = LogLawWind(6.0, 20.0, z0=0.5, displacement=10.0, direction_deg=90.0)
    out = sample(w, [[0, 0, 20.0], [0, 0, 40.0]])
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12) and out[0, 1] == pytest.approx(6.0)  # blowing towards +y
    assert out[1, 1] > out[0, 1]
    low = sample(w, [[0, 0, 0.0], [0, 0, 11.0], [0, 0, 5.0]])
    assert np.all(np.isfinite(low)) and np.allclose(low[0], low[2])  # held constant inside the roughness layer


def test_log_law_validation_and_roundtrip():
    with pytest.raises(ValueError):
        LogLawWind(z0=0.0)
    with pytest.raises(ValueError):
        LogLawWind(height_ref=1.0, displacement=1.0)
    w = LogLawWind(7.0, 12.0, 0.2, 3.0, 45.0)
    assert wind_from_dict(w.to_dict()).to_dict() == w.to_dict()


# ----------------------------------------------------------------------------------------------- gridded
def linear_field(nx=5, ny=6, nz=7, spacing=(2.0, 1.5, 1.0), origin=(-3.0, 2.0, 0.0)):
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    x, y, z = origin[0] + ix * spacing[0], origin[1] + iy * spacing[1], origin[2] + iz * spacing[2]
    data = np.stack([1.0 + 0.5 * x - 0.2 * y + 0.1 * z, -2.0 + 0.3 * y + 0.05 * z, 0.4 * z - 0.1 * x], -1)
    return data, np.array(origin), np.array(spacing)


def test_gridded_wind_is_exact_at_nodes_and_reproduces_linear_fields_everywhere():
    data, origin, spacing = linear_field()
    w = GriddedWind(data, origin, spacing)
    rng = np.random.default_rng(0)
    p = origin + rng.uniform(0, 1, (200, 3)) * (np.array(data.shape[:3]) - 1) * spacing
    x, y, z = p.T
    exact = np.stack([1.0 + 0.5 * x - 0.2 * y + 0.1 * z, -2.0 + 0.3 * y + 0.05 * z, 0.4 * z - 0.1 * x], -1)
    np.testing.assert_allclose(sample(w, p), exact, atol=1e-12)
    nodes = origin + np.array([[0, 0, 0], [4, 5, 6], [2, 3, 1]]) * spacing
    np.testing.assert_allclose(sample(w, nodes), data[[0, 4, 2], [0, 5, 3], [0, 6, 1]], atol=1e-12)


def test_outside_the_grid_holds_the_boundary_or_is_calm():
    data, origin, spacing = linear_field()
    far = [[1e4, 1e4, 1e4], [-1e4, 0, 0]]
    held = sample(GriddedWind(data, origin, spacing, outside="nearest"), far)
    assert np.allclose(held[0], data[-1, -1, -1]) and np.all(np.isfinite(held))
    assert np.allclose(sample(GriddedWind(data, origin, spacing, outside="zero"), far), 0.0)
    assert np.allclose(sample(GriddedWind(data, origin, spacing, scale=2.0), [origin + spacing]), 2.0 * data[1, 1, 1])


def test_time_dependent_grid_interpolates_linearly_clamps_and_loops():
    base, origin, spacing = linear_field(3, 3, 3)
    frames = np.stack([base * 0.0, base * 1.0, base * 3.0])
    w = GriddedWind(frames, origin, spacing, times=[0.0, 1.0, 3.0])
    p = [origin + spacing]
    ref = base[1, 1, 1]
    assert np.allclose(sample(w, p, 0.5)[0], 0.5 * ref) and np.allclose(sample(w, p, 2.0)[0], 2.0 * ref)
    assert np.allclose(sample(w, p, -5.0)[0], 0.0) and np.allclose(sample(w, p, 99.0)[0], 3.0 * ref)
    loop = GriddedWind(frames, origin, spacing, times=[0.0, 1.0, 3.0], loop=True)
    assert np.allclose(sample(loop, p, 3.5)[0], sample(loop, p, 0.5)[0])


def test_grid_validation():
    data, origin, spacing = linear_field()
    for kwargs in (
        dict(data=data[..., :2]),
        dict(data=data, spacing=0.0),
        dict(data=data, outside="mirror"),
        dict(data=np.where(data > 0, np.nan, data)),
    ):
        kw = dict(data=data, origin=origin, spacing=spacing)
        kw.update(kwargs)
        with pytest.raises(ValueError):
            GriddedWind(**kw)
    with pytest.raises(ValueError):
        GriddedWind(data, origin, spacing, times=[0.0, 1.0])
    with pytest.raises(ValueError):
        GriddedWind(np.stack([data, data]), origin, spacing, times=[1.0, 1.0])
    with pytest.raises(ValueError):
        GriddedWind(data[:1], origin, spacing)


def test_npz_round_trip_and_serialisation(tmp_path):
    data, origin, spacing = linear_field()
    w = GriddedWind(data, origin, spacing)
    with pytest.raises(ValueError, match="source"):
        w.to_dict()
    path = w.save_npz(tmp_path / "wind.npz")
    again = wind_from_dict(w.to_dict())
    pts = origin + np.array([[1.3, 0.7, 2.2]]) * spacing
    np.testing.assert_allclose(sample(again, pts), sample(w, pts))
    assert GriddedWind.from_npz(path).extent.tolist() == w.extent.tolist()
    np.savez(tmp_path / "bad.npz", u=data)
    with pytest.raises(ValueError, match="missing"):
        GriddedWind.from_npz(tmp_path / "bad.npz")


def test_csv_import_any_row_order_and_irregular_grids_are_rejected(tmp_path):
    data, origin, spacing = linear_field(3, 4, 2)
    rows = []
    for i in range(3):
        for j in range(4):
            for k in range(2):
                rows.append([*(origin + np.array([i, j, k]) * spacing), *data[i, j, k]])
    rows = np.array(rows)
    np.random.default_rng(3).shuffle(rows)
    p = tmp_path / "w.csv"
    np.savetxt(p, rows, delimiter=",", header="x,y,z,u,v,w", comments="")
    w = GriddedWind.from_csv(p)
    np.testing.assert_allclose(w.data[0], data)
    bad = rows.copy()
    bad[0, 0] += 0.3
    np.savetxt(tmp_path / "bad.csv", bad, delimiter=",", header="x,y,z,u,v,w", comments="")
    with pytest.raises(ValueError):
        GriddedWind.from_csv(tmp_path / "bad.csv")
    np.savetxt(tmp_path / "short.csv", rows[:-1], delimiter=",", header="x,y,z,u,v,w", comments="")
    with pytest.raises(ValueError, match="regular"):
        GriddedWind.from_csv(tmp_path / "short.csv")


def test_divergence_of_a_solenoidal_field_is_zero():
    n = 12
    ix, iy, iz = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    rot = np.stack([iy * 1.0, -ix * 1.0, np.zeros_like(ix, float)], -1)  # solid-body rotation: div = 0
    assert np.allclose(GriddedWind(rot, (0, 0, 0), 1.0).divergence(), 0.0, atol=1e-12)
    expanding = np.stack([ix * 1.0, iy * 1.0, iz * 1.0], -1)
    assert np.allclose(GriddedWind(expanding, (0, 0, 0), 1.0).divergence(), 3.0)


# -------------------------------------------------------------------------------------------------- urban
def test_fackrell_length_matches_the_published_correlation():
    h, w, l = 10.0, 20.0, 15.0
    assert fackrell_reattachment_length(h, w, l) == pytest.approx(10.0 * 1.8 * 2.0 / (1.5**0.3 * (1 + 0.24 * 2.0)))
    assert fackrell_reattachment_length(10, 40, 15) > fackrell_reattachment_length(10, 10, 15)  # wider buildings: longer wake
    assert fackrell_reattachment_length(10, 20, 60) < fackrell_reattachment_length(10, 20, 10)  # longer buildings: shorter cavity


def urban(direction=0.0, **kw):
    box = [[-10.0, -10.0, 0.0, 10.0, 10.0, 20.0]]  # 20 x 20 x 20 m building at the origin
    return UrbanWind(speed_ref=8.0, height_ref=10.0, z0=0.5, direction_deg=direction, buildings=box, **kw)


def test_far_from_buildings_the_wind_is_exactly_the_ambient_profile():
    w = urban()
    p = np.array([[500.0, 0, 15.0], [0, 400.0, 30.0], [-300.0, -300.0, 5.0], [200.0, 0, 200.0]])
    np.testing.assert_allclose(sample(w, p), sample(LogLawWind(8.0, 10.0, 0.5), p), atol=1e-9)


def test_cavity_reverses_the_flow_and_it_recovers_downwind():
    w = urban()
    lr = w.cavity_lengths()[0]
    assert lr == pytest.approx(fackrell_reattachment_length(20, 20, 20))
    x = np.array([10.5, 10.0 + 0.5 * lr, 10.0 + lr, 10.0 + 2.5 * lr, 10.0 + 10 * lr])
    out = sample(w, [[xx, 0.0, 8.0] for xx in x])
    assert out[0, 0] < 0.0  # reversed flow right behind the lee face
    assert out[2, 0] == pytest.approx(0.0, abs=0.15 * 8.0)  # ~zero at the reattachment point
    assert np.all(np.diff(out[:, 0]) > 0)  # monotone recovery
    ambient = float(LogLawWind(8.0, 10.0, 0.5).speed_at(8.0))
    assert out[-1, 0] == pytest.approx(ambient, rel=0.02)
    assert out[1, 2] < 0.0  # downwash inside the cavity


def test_windward_side_slows_and_lifts_the_flow():
    w = urban()
    out = sample(w, [[-12.0, 0.0, 8.0], [-150.0, 0.0, 8.0]])
    ambient = float(LogLawWind(8.0, 10.0, 0.5).speed_at(8.0))
    assert out[0, 0] < ambient and out[0, 0] > 0.4 * ambient and out[0, 2] > 0.0
    assert out[1, 0] == pytest.approx(ambient, rel=1e-3)


def test_wake_is_narrow_and_symmetric_and_vanishes_above_the_roof():
    w = urban()
    ambient = float(LogLawWind(8.0, 10.0, 0.5).speed_at(8.0))
    lr = w.cavity_lengths()[0]
    xs = 10.0 + 0.3 * lr
    left, right = sample(w, [[xs, 6.0, 8.0], [xs, -6.0, 8.0]])
    np.testing.assert_allclose(left, right, atol=1e-12)
    far_side = sample(w, [[xs, 80.0, 8.0]])[0]
    assert far_side[0] == pytest.approx(ambient, rel=1e-3)
    above = sample(w, [[xs, 0.0, 120.0]])[0]
    assert above[0] == pytest.approx(float(LogLawWind(8.0, 10.0, 0.5).speed_at(120.0)), rel=1e-3)


def test_field_is_invariant_under_rotation_of_wind_and_world():
    """Turning the wind by 90 degrees and the point of view with it gives the same wake."""
    a = urban(0.0)
    b = urban(90.0)  # a square building rotated onto itself
    pa = np.array([[25.0, 3.0, 9.0], [-15.0, -4.0, 5.0], [14.0, 0.0, 12.0]])
    rot = np.array([[0, -1], [1, 0]])
    pb = np.column_stack([pa[:, :2] @ rot.T, pa[:, 2]])
    wa, wb = sample(a, pa), sample(b, pb)
    np.testing.assert_allclose(wb[:, :2], wa[:, :2] @ rot.T, atol=1e-9)
    np.testing.assert_allclose(wb[:, 2], wa[:, 2], atol=1e-9)


def test_field_is_continuous_bounded_and_finite():
    w = urban(30.0)
    rng = np.random.default_rng(4)
    p = rng.uniform([-60, -60, 0.5], [120, 60, 60], (4000, 3))
    out = sample(w, p)
    assert np.all(np.isfinite(out))
    ambient = LogLawWind(8.0, 10.0, 0.5, direction_deg=30.0).speed_at(p[:, 2])
    assert np.all(np.linalg.norm(out[:, :2], axis=1) <= 1.01 * ambient)  # buildings only ever slow the horizontal wind
    d = 0.02
    steps = rng.normal(size=(4000, 3))
    steps = d * steps / np.linalg.norm(steps, axis=1, keepdims=True)
    jump = np.linalg.norm(sample(w, p + steps) - out, axis=1).max()
    assert jump < 0.1  # Lipschitz-like: 2 cm steps change the wind by < 0.1 m/s


def test_strength_zero_and_small_buildings_are_ignored():
    p = [[12.0, 0.0, 5.0]]
    off = UrbanWind(8.0, 10.0, 0.5, buildings=[[-10, -10, 0, 10, 10, 20]], strength=0.0)
    np.testing.assert_allclose(sample(off, p), sample(LogLawWind(8.0, 10.0, 0.5), p))
    tiny = UrbanWind(8.0, 10.0, 0.5, buildings=[[-1, -1, 0, 1, 1, 1.0]])
    assert tiny.n_buildings == 0


def test_urban_wind_takes_buildings_from_the_world_and_composes():
    world = World.city(blocks=(2, 2), seed=1)
    world.wind = CompositeWind(UrbanWind(8.0, 10.0, 0.5), ConstantWind([0.0, 1.0, 0.0]))
    world.reset(B, 1, 0)
    u = world.wind.models[0]
    assert u.n_buildings == len(world.boxes)
    out = np.asarray(world.wind.sample(B, np.array([[0.0, 0.0, 30.0]]), np.zeros((1, 3)), 0.0, 0.01))
    assert out.shape == (1, 3) and np.all(np.isfinite(out))
    assert wind_from_dict(UrbanWind(8, 10, 0.5, buildings=[[0, 0, 0, 5, 5, 9]]).to_dict()).n_buildings == 1


def test_registry_and_world_serialisation(tmp_path):
    from aevion.registry import registries

    assert {"log_law", "urban", "gridded"} <= set(registries.winds.names())
    data, origin, spacing = linear_field()
    GriddedWind(data, origin, spacing).save_npz(tmp_path / "g.npz")
    w = registries.winds.create("gridded", file=str(tmp_path / "g.npz"))
    assert isinstance(w, GriddedWind)
    from aevion.errors import RegistryError

    with pytest.raises(RegistryError, match="needs 'file'"):
        registries.winds.create("gridded")
    world = World()
    world.wind = w
    assert World.from_dict(world.to_dict()).wind.source == str(tmp_path / "g.npz")
