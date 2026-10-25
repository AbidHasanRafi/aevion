"""Geodesy, geomagnetic field, solar ephemeris and weather, checked against textbook geometry and published values."""

import datetime as dt
import math

import numpy as np
import pytest

from aevion.core.backend import NumpyBackend
from aevion.core.constants import Atmosphere, saturation_vapour_pressure
from aevion.environments import geodesy as G
from aevion.environments import solar as S
from aevion.environments.geomagnetic import FixedField, TiltedDipole, field_angles
from aevion.environments.weather import PRESETS, Weather, fog_extinction, moist_air_density, rain_extinction, visibility_from_extinction
from aevion.environments.world import World

B = NumpyBackend()


# ------------------------------------------------------------------------------------------------ geodesy
def test_ecef_of_the_equator_and_the_pole():
    np.testing.assert_allclose(G.lla_to_ecef(0, 0, 0), [G.WGS84_A, 0, 0], atol=1e-9)
    np.testing.assert_allclose(G.lla_to_ecef(90, 0, 0), [0, 0, G.WGS84_B], atol=1e-6)
    np.testing.assert_allclose(G.lla_to_ecef(0, 90, 100), [0, G.WGS84_A + 100, 0], atol=1e-9)


def test_lla_ecef_round_trip_everywhere_including_poles_and_extreme_altitudes():
    rng = np.random.default_rng(0)
    lat = np.concatenate([rng.uniform(-90, 90, 500), [90.0, -90.0, 0.0, 89.9999, -89.9999]])
    lon = np.concatenate([rng.uniform(-180, 180, 500), [0.0, 10.0, 180.0, -179.0, 45.0]])
    alt = np.concatenate([rng.uniform(-10_000, 100_000, 500), [0.0, -100.0, 5e4, 1.0, 20000.0]])
    back = G.ecef_to_lla(G.lla_to_ecef(lat, lon, alt))
    np.testing.assert_allclose(back[:, 0], lat, atol=1e-9)
    np.testing.assert_allclose(back[:, 2], alt, atol=1e-6)
    dlon = (back[:, 1] - lon + 180) % 360 - 180
    polar = np.abs(lat) > 89.99
    np.testing.assert_allclose(dlon[~polar], 0, atol=1e-9)


def test_local_frame_is_a_right_handed_orthonormal_basis_aligned_with_the_ellipsoid_normal():
    f = G.LocalFrame(47.3769, 8.5417, 408.0)
    np.testing.assert_allclose(f.rotation @ f.rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(f.rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(f.to_enu([47.3769, 8.5417, 508.0]), [0, 0, 100.0], atol=1e-6)  # straight up is 'up'
    np.testing.assert_allclose(f.to_enu([47.3769, 8.5417, 408.0]), 0, atol=1e-6)


def test_enu_distances_match_the_radii_of_curvature():
    f = G.LocalFrame(0.0, 0.0, 0.0)
    north = f.to_enu([0.01, 0, 0])  # small offsets: the local tangent plane agrees with the arc length
    assert north[1] == pytest.approx(math.radians(0.01) * G.meridian_radius(0.0), abs=0.01)  # 110.574 km per degree at the equator
    east = f.to_enu([0, 0.01, 0])
    assert east[0] == pytest.approx(math.radians(0.01) * G.prime_vertical_radius(0.0), abs=0.01)  # 111.319 km
    f60 = G.LocalFrame(60.0, 10.0, 0.0)
    assert f60.to_enu([60.0, 10.01, 0.0])[0] == pytest.approx(
        math.radians(0.01) * G.prime_vertical_radius(60.0) * math.cos(math.radians(60.0)), rel=2e-4
    )


def test_enu_round_trip():
    f = G.LocalFrame(-33.9, 151.2, 20.0)
    rng = np.random.default_rng(1)
    enu = rng.uniform(-5000, 5000, (50, 3))
    np.testing.assert_allclose(f.to_enu(f.to_lla(enu)), enu, atol=1e-6)
    assert G.LocalFrame.from_dict(f.to_dict()).to_dict() == f.to_dict()
    with pytest.raises(ValueError):
        G.LocalFrame(95.0, 0.0)


def test_vincenty_reproduces_the_geoscience_australia_test_line():
    """Flinders Peak -> Buninyong, the standard Vincenty example: 54 972.271 m, azimuths 306 52' 05.37'' and 127 10' 25.07''."""
    lat1, lon1 = -(37 + 57 / 60 + 3.72030 / 3600), 144 + 25 / 60 + 29.52440 / 3600
    lat2, lon2 = -(37 + 39 / 60 + 10.15610 / 3600), 143 + 55 / 60 + 35.38390 / 3600
    d, a1, a2 = G.vincenty_inverse(lat1, lon1, lat2, lon2)
    assert d == pytest.approx(54972.271, abs=0.005)
    assert a1 == pytest.approx(306 + 52 / 60 + 5.37 / 3600, abs=1e-5)
    assert a2 == pytest.approx(127 + 10 / 60 + 25.07 / 3600, abs=1e-5)


def test_vincenty_direct_inverts_the_inverse():
    rng = np.random.default_rng(2)
    for _ in range(25):
        lat1, lat2 = rng.uniform(-80, 80, 2)
        lon1, lon2 = rng.uniform(-170, 170, 2)
        d, az1, _ = G.vincenty_inverse(lat1, lon1, lat2, lon2)
        if math.isnan(d):
            continue
        la, lo, _ = G.vincenty_direct(lat1, lon1, az1, d)
        assert la == pytest.approx(lat2, abs=1e-8) and (lo - lon2 + 180) % 360 - 180 == pytest.approx(0, abs=1e-8)


def test_equator_arc_and_degenerate_geodesics():
    assert G.vincenty_inverse(0, 0, 0, 1)[0] == pytest.approx(math.radians(1) * G.WGS84_A, abs=1e-4)
    assert G.vincenty_inverse(10, 20, 10, 20) == (0.0, 0.0, 0.0)
    assert math.isnan(G.vincenty_inverse(0, 0, 0.5, 179.7)[0])  # near-antipodal: no convergence, reported as nan not a wrong number


def test_haversine_and_bearing():
    assert G.haversine_distance(0, 0, 90, 0) == pytest.approx(math.pi / 2 * 6371008.8, rel=1e-9)
    d, _, _ = G.vincenty_inverse(47.37, 8.54, 48.85, 2.35)  # Zurich - Paris
    assert G.haversine_distance(47.37, 8.54, 48.85, 2.35) == pytest.approx(d, rel=6e-3)
    assert G.initial_bearing(0, 0, 0, 10) == pytest.approx(90.0)
    assert G.initial_bearing(0, 0, 10, 0) == pytest.approx(0.0)
    assert G.initial_bearing(10, 0, 0, 0) == pytest.approx(180.0)


# ------------------------------------------------------------------------------------------- geomagnetic
def test_tilted_dipole_reproduces_the_igrf_degree_one_pole_and_moment():
    m = TiltedDipole()
    lat, lon = m.geomagnetic_pole()
    assert lat == pytest.approx(80.6, abs=0.1) and lon == pytest.approx(-72.7, abs=0.1)
    assert m.moment_nt == pytest.approx(29806, abs=2)
    inc = m.angles(lat, lon, 0.0)["inclination_deg"]
    assert inc == pytest.approx(90.0, abs=1.0)  # field points straight down at the pole (ellipsoid shifts it slightly)


def test_dipole_inclination_follows_tan_i_equals_two_tan_geomagnetic_latitude():
    m = TiltedDipole()
    plat, plon = np.radians(m.geomagnetic_pole())
    rng = np.random.default_rng(3)
    for _ in range(30):
        lat, lon = rng.uniform(-75, 75), rng.uniform(-180, 180)
        sin_gm = math.sin(math.radians(lat)) * math.sin(plat) + math.cos(math.radians(lat)) * math.cos(plat) * math.cos(math.radians(lon) - plon)
        gm = math.asin(sin_gm)
        inc = float(m.angles(lat, lon, 0.0)["inclination_deg"])
        assert inc == pytest.approx(math.degrees(math.atan(2 * math.tan(gm))), abs=1.6)  # exact for a geocentric dipole; ellipsoid adds < 1.6 deg


def test_dipole_intensity_is_twice_as_strong_at_the_pole_as_at_the_magnetic_equator_and_falls_as_r_cubed():
    m = TiltedDipole()
    lats, lons = np.meshgrid(np.linspace(-90, 90, 91), np.linspace(-180, 180, 181))
    inten = m.angles(lats, lons, 0.0)["intensity_ut"]
    assert inten.max() / inten.min() == pytest.approx(2.0, rel=0.03)
    surface, high = m.angles(30, 40, 0.0)["intensity_ut"], m.angles(30, 40, 400_000.0)["intensity_ut"]
    r0, r1 = np.linalg.norm(G.lla_to_ecef(30, 40, 0.0)), np.linalg.norm(G.lla_to_ecef(30, 40, 400_000.0))
    assert high / surface == pytest.approx((r0 / r1) ** 3, rel=5e-4)  # exact up to the ellipsoid's geodetic/geocentric tilt


def test_dipole_secular_variation_and_shapes():
    m = TiltedDipole()
    assert not np.allclose(m.field_enu(45, 10, 0, 2020.0), m.field_enu(45, 10, 0, 2030.0))
    assert m.field_enu(np.zeros((4, 5)), np.zeros((4, 5)), 0.0).shape == (4, 5, 3)
    assert m.to_dict()["type"] == "TiltedDipole"


def test_fixed_field_matches_the_legacy_magnetometer_convention():
    from aevion.sensors import Magnetometer

    f = FixedField(intensity_ut=48.0, inclination_deg=63.0, declination_deg=3.0)
    np.testing.assert_allclose(
        f.field_enu(10, 20, 30), Magnetometer(field_ut=48.0, inclination_deg=63.0, declination_deg=3.0).field_world, atol=1e-12
    )
    a = field_angles(f.field_enu(0, 0, 0))
    assert a["inclination_deg"] == pytest.approx(63.0) and a["declination_deg"] == pytest.approx(3.0) and a["intensity_ut"] == pytest.approx(48.0)


# ----------------------------------------------------------------------------------------------- solar
@pytest.mark.parametrize(
    "when,decl,tol",
    [
        ("2026-03-20T12:00:00Z", 0.0, 0.15),
        ("2026-06-21T12:00:00Z", 23.44, 0.03),
        ("2026-12-21T12:00:00Z", -23.44, 0.03),
        ("2026-09-23T12:00:00Z", -0.19, 0.03),
    ],
)
def test_solar_declination_at_equinoxes_and_solstices(when, decl, tol):
    assert float(S.sun_position(0, 0, when)["declination_deg"]) == pytest.approx(decl, abs=tol)


@pytest.mark.parametrize(
    "when,eot", [("2026-11-03T12:00:00Z", 16.4), ("2026-02-11T12:00:00Z", -14.2), ("2026-04-15T12:00:00Z", 0.0), ("2026-12-25T12:00:00Z", 0.0)]
)
def test_equation_of_time_matches_published_extremes(when, eot):
    assert float(S.equation_of_time_min(when)) == pytest.approx(eot, abs=0.4)


def test_solar_noon_elevation_is_ninety_minus_latitude_plus_declination():
    times = np.datetime64("2026-06-21T00:00:00") + np.arange(0, 1440) * np.timedelta64(60, "s")
    for lat, lon in [(40.0, 0.0), (-33.9, 151.2), (0.0, 30.0)]:
        el = S.sun_position(lat, lon, times)["elevation_deg"]
        expect = 90.0 - abs(lat - 23.44)
        assert float(el.max()) == pytest.approx(expect, abs=0.15)


def test_sun_rises_in_the_east_and_is_south_at_noon_in_the_northern_hemisphere():
    t = np.datetime64("2026-03-20T00:00:00") + np.arange(0, 1440, 5) * np.timedelta64(60, "s")
    p = S.sun_position(45.0, 0.0, t)
    up = p["elevation_deg"] > 0
    first, noon = np.nonzero(up)[0][0], int(np.argmax(p["elevation_deg"]))
    assert 80 < p["azimuth_deg"][first] < 100  # equinox: due east at sunrise
    assert p["azimuth_deg"][noon] == pytest.approx(180.0, abs=2.0)
    assert np.all(np.diff(p["azimuth_deg"][up]) > 0)  # moves clockwise


def test_sunrise_and_sunset_and_polar_cases():
    rise, sett = S.sunrise_sunset(47.37, 8.54, "2026-06-21")
    assert (sett - rise).total_seconds() / 3600 == pytest.approx(15.95, abs=0.1)  # Zurich: ~15 h 57 min
    assert rise.hour == 3 and 20 <= rise.minute <= 40  # 05:29 CEST
    r2, s2 = S.sunrise_sunset(0.0, 0.0, "2026-03-20")
    assert (s2 - r2).total_seconds() / 3600 == pytest.approx(12.1, abs=0.1)
    assert S.sunrise_sunset(80.0, 0.0, "2026-06-21") == (None, None)  # midnight sun
    assert S.sunrise_sunset(80.0, 0.0, "2026-12-21") == (None, None)  # polar night


def test_time_inputs_are_interchangeable():
    iso = S.sun_position(10, 20, "2026-05-01T09:30:00Z")["elevation_deg"]
    naive = S.sun_position(10, 20, dt.datetime(2026, 5, 1, 9, 30))["elevation_deg"]
    d64 = S.sun_position(10, 20, np.datetime64("2026-05-01T09:30:00"))["elevation_deg"]
    unix = S.sun_position(10, 20, dt.datetime(2026, 5, 1, 9, 30, tzinfo=dt.timezone.utc).timestamp())["elevation_deg"]
    assert float(iso) == pytest.approx(float(naive)) == pytest.approx(float(d64)) == pytest.approx(float(unix))
    assert float(S.julian_day("2000-01-01T12:00:00Z")) == pytest.approx(2451545.0)


def test_sun_direction_and_irradiance():
    np.testing.assert_allclose(S.sun_direction_enu(90.0, 0.0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(S.sun_direction_enu(0.0, 90.0), [0, 0, 1], atol=1e-12)
    assert float(S.clear_sky_ghi(90.0)) == pytest.approx(1098 * math.exp(-0.057), rel=1e-9)
    assert float(S.clear_sky_ghi(-5.0)) == 0.0
    ghi = S.clear_sky_ghi(np.linspace(1, 90, 50))
    assert np.all(np.diff(ghi) > 0)
    assert float(S.refracted_elevation(0.0)) == pytest.approx(0.483, abs=0.05) and float(S.refracted_elevation(60.0)) == pytest.approx(60.0, abs=0.02)


# ---------------------------------------------------------------------------------------------- weather
def test_default_atmosphere_is_exactly_the_isa_and_weather_is_opt_in():
    z = np.array([0.0, 500.0, 3000.0])
    a = Atmosphere()
    assert np.array_equal(a.density(B, z), a.pressure(B, z) / (287.05287 * a.temperature(B, z)))
    assert a.to_dict() == dict(base_altitude=0.0, temp_offset=0.0, enabled=True)


def test_moist_air_density_physics():
    assert moist_air_density(101325.0, 15.0, 0.0) == pytest.approx(1.2250, abs=5e-4)  # ISA sea level
    dry, wet = moist_air_density(101325.0, 30.0, 0.0), moist_air_density(101325.0, 30.0, 1.0)
    assert wet < dry and (dry - wet) / dry == pytest.approx(0.0165, abs=0.004)  # humid air is lighter
    assert saturation_vapour_pressure(20.0) == pytest.approx(2339.0, rel=0.005)
    assert saturation_vapour_pressure(100.0) == pytest.approx(101325.0, rel=0.04)


def test_weather_changes_the_world_atmosphere_consistently():
    w = World()
    wx = Weather.preset("hot")
    wx.apply(w)
    rho = float(w.atmosphere.density(B, np.array([0.0]))[0])
    assert rho == pytest.approx(wx.air_density(), rel=1e-9)
    assert rho < 1.2250 * 0.93  # hot, dry-ish air at sea level is >7 % thinner than ISA
    low = Weather(pressure_sl_pa=95_000.0).air_density()
    assert low < Weather(pressure_sl_pa=101_325.0).air_density()
    assert W_round_trip(w).weather.to_dict() == wx.to_dict()


def W_round_trip(world):
    return World.from_dict(world.to_dict())


def test_visibility_and_extinction():
    assert fog_extinction(1000.0) == pytest.approx(3.912e-3)
    assert visibility_from_extinction(fog_extinction(250.0)) == pytest.approx(250.0)
    assert rain_extinction(0.0) == 0.0
    assert rain_extinction(1.0) == pytest.approx(0.365e-3)  # Marshall-Palmer geometric optics: 0.365 R^0.63 per km
    assert rain_extinction(10.0) == pytest.approx(0.365e-3 * 10**0.63)
    w = Weather(visibility_m=2000.0, rain_mm_h=10.0)
    assert w.extinction() == pytest.approx(fog_extinction(2000.0) + rain_extinction(10.0))
    assert w.effective_visibility_m < 2000.0
    assert w.transmission(100.0) == pytest.approx(math.exp(-w.extinction() * 100.0))


def test_marshall_palmer_extinction_matches_its_defining_integral():
    """beta = (pi/2) * integral N(D) D^2 dD with N = 8000 exp(-4.1 R^-0.21 D) reproduces 0.365 R^0.63 / km."""
    for r in (1.0, 5.0, 25.0):
        lam = 4.1 * r**-0.21
        d = np.linspace(0, 12.0 / lam, 200_001)
        n = 8000.0 * np.exp(-lam * d)  # per m^3 per mm
        beta = 0.5 * math.pi * np.trapezoid(n * d * d, d) * 1e-6  # mm^2 -> m^2, result per metre
        assert beta == pytest.approx(rain_extinction(r), rel=2e-3)


def test_presets_and_validation():
    for name in PRESETS:
        w = Weather.preset(name)
        assert w.air_density() > 0.9 and w.extinction() > 0
    assert Weather.preset("fog").effective_visibility_m < 250
    assert Weather.from_dict({"preset": "rain", "rain_mm_h": 8.0}).rain_mm_h == 8.0
    for bad in (dict(humidity=1.5), dict(cloud_cover=-0.1), dict(rain_mm_h=-1), dict(visibility_m=0)):
        with pytest.raises(ValueError):
            Weather(**bad)
    with pytest.raises(ValueError, match="unknown weather"):
        Weather.from_dict({"temprature_c": 5})
    with pytest.raises(KeyError):
        Weather.preset("hail")
