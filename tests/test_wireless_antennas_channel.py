"""Antenna patterns/arrays/beamforming/attitude and channel models (MIMO, blockage, delay spread, kinematics)."""

import math

import numpy as np
import pytest

from aevion.wireless.antennas import (
    GAIN_FLOOR_DBI,
    BeamManager,
    GaussianBeam,
    HalfWaveDipole,
    Isotropic,
    Parabolic,
    Patch3GPP,
    SectorTR36814,
    ShortDipole,
    UavAntenna,
    UniformLinearArray,
    UniformPlanarArray,
    dft_codebook,
    direction_in_antenna_frame,
    direction_vector,
    grid_codebook,
    mmse_weights,
    mrt_weights,
    off_boresight_angle,
    pointing_loss_db,
    quat_to_matrices,
    rpy_to_matrix,
    zf_weights,
)
from aevion.wireless.channel import (
    BlockageProcess,
    blockage_probability,
    blockage_rate,
    coherence_bandwidth_hz,
    condition_number_db,
    delay_spread_params,
    ergodic_capacity,
    exponential_correlation,
    exponential_pdp,
    frequency_response,
    geometric_channel,
    link_kinematics,
    mimo_capacity,
    rayleigh_channel,
    rms_delay_spread,
    sample_delay_spread_s,
    self_blockage_loss_db,
    water_filling_capacity,
)


def rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------------------------------- patterns
def test_pattern_peak_gains_match_theory():
    assert Isotropic(3.0).peak_dbi == pytest.approx(3.0)
    assert ShortDipole().peak_dbi == pytest.approx(10 * math.log10(1.5), abs=0.01)  # 1.76 dBi
    assert HalfWaveDipole().peak_dbi == pytest.approx(2.15, abs=0.02)
    assert Patch3GPP().peak_dbi == pytest.approx(8.0)
    assert Patch3GPP(g_max_dbi=5.0).peak_dbi == pytest.approx(5.0)


def test_dipole_has_a_null_on_its_axis_and_is_omnidirectional_in_azimuth():
    for d in (ShortDipole(), HalfWaveDipole()):
        assert float(d.gain_dbi(0.3, math.pi / 2)) <= -30.0  # along the dipole axis (up)
        az = np.linspace(-math.pi, math.pi, 13)
        g = d.gain_dbi(az, 0.0)
        assert np.allclose(g, g[0])


def test_patch_pattern_follows_the_3gpp_roll_off():
    p = Patch3GPP()
    assert float(p.gain_dbi(0.0, 0.0)) == pytest.approx(8.0)
    az = math.radians(65.0)  # 12 (65/65)^2 = 12 dB down at the 3 dB... (3GPP definition)
    assert float(p.gain_dbi(az, 0.0)) == pytest.approx(8.0 - 12.0)
    assert float(p.gain_dbi(math.pi, 0.0)) == pytest.approx(8.0 - 30.0)  # back lobe capped at A_max
    assert float(p.gain_dbi(0.0, math.radians(90.0))) >= GAIN_FLOOR_DBI


def test_sector_matches_the_legacy_vertical_pattern_on_the_azimuth_boresight():
    from aevion.wireless.channel import sector_gain_dbi

    s = SectorTR36814(g_max_dbi=8.0, tilt_deg=10.0)
    for below in (-20.0, 0.0, 10.0, 40.0):
        assert float(s.gain_dbi(0.0, math.radians(-below))) == pytest.approx(float(sector_gain_dbi(below, 10.0, g_max=8.0)), abs=1e-9)


def test_gaussian_and_parabolic_beams():
    g = GaussianBeam(g_max_dbi=15.0, bw_az_deg=30.0, bw_el_deg=10.0)
    assert float(g.gain_dbi(0.0, 0.0)) == pytest.approx(15.0)
    assert float(g.gain_dbi(math.radians(15.0), 0.0)) == pytest.approx(15.0 - 3.0)  # 12 (15/30)^2 = 3 dB at the half beamwidth
    assert float(g.gain_dbi(0.0, math.radians(5.0))) == pytest.approx(15.0 - 3.0)
    dish = Parabolic(diameter_m=0.3, freq_hz=28e9, efficiency=0.6)
    lam = 299_792_458.0 / 28e9
    assert dish.peak_dbi == pytest.approx(10 * math.log10(0.6 * (math.pi * 0.3 / lam) ** 2), abs=1e-6)
    assert float(dish.gain_dbi(math.radians(dish.bw / 2), 0.0)) == pytest.approx(dish.peak_dbi - 3.0, abs=0.01)


def test_off_boresight_angle_geometry():
    assert float(off_boresight_angle(0.0, 0.0)) == 0.0
    assert float(off_boresight_angle(math.pi / 2, 0.0)) == pytest.approx(math.pi / 2)
    assert float(off_boresight_angle(0.0, math.pi / 2)) == pytest.approx(math.pi / 2)
    assert float(off_boresight_angle(math.pi, 0.0)) == pytest.approx(math.pi)


# --------------------------------------------------------------------------------------------- arrays
def test_steered_ula_reaches_the_array_gain_and_the_theoretical_beamwidth():
    ula = UniformLinearArray(8, element=Isotropic(0.0))
    az = math.radians(25.0)
    w = ula.beam_weights(az, 0.0)
    assert float(np.linalg.norm(w)) == pytest.approx(1.0)
    assert float(ula.array_factor_db(w, az, 0.0)) == pytest.approx(10 * math.log10(8), abs=1e-9)
    assert float(ula.gain_dbi(w, az, 0.0)) == pytest.approx(10 * math.log10(8), abs=1e-9)
    # broadside half-power beamwidth ~ 0.886 lambda / (N d) rad
    assert ula.half_power_beamwidth_deg() == pytest.approx(math.degrees(0.886 / (8 * 0.5)), rel=0.02)
    # the beam broadens when steered off broadside (cos factor)
    assert ula.half_power_beamwidth_deg(steer_deg=45.0) > ula.half_power_beamwidth_deg()
    # nulls: a direction a bit off has far lower gain
    assert float(ula.array_factor_db(w, az + math.radians(30), 0.0)) < 0.0


def test_upa_has_n_elements_and_steers_in_elevation():
    upa = UniformPlanarArray(4, 4, element=Isotropic(0.0))
    assert upa.n_elements == 16
    w = upa.beam_weights(0.0, math.radians(20.0))
    assert float(upa.array_factor_db(w, 0.0, math.radians(20.0))) == pytest.approx(10 * math.log10(16), abs=1e-9)
    assert float(upa.array_factor_db(w, 0.0, 0.0)) < 10 * math.log10(16) - 3
    with pytest.raises(ValueError):
        UniformPlanarArray(0, 3)
    with pytest.raises(ValueError):
        UniformLinearArray(0)


def test_direction_vector_is_a_unit_vector():
    u = direction_vector(np.array([0.3, -1.2]), np.array([0.2, 0.9]))
    assert np.allclose(np.linalg.norm(u, axis=-1), 1.0)


def test_dft_codebook_covers_the_field_of_view_with_small_scalloping_loss():
    ula = UniformLinearArray(8, element=Isotropic(0.0))
    cb = dft_codebook(ula, oversample=2, fov_deg=120.0)
    assert len(cb) >= 8
    az = np.radians(np.linspace(-55, 55, 111))
    idx, gain = cb.best_beam(ula, az, np.zeros_like(az))
    assert gain.min() > 10 * math.log10(8) - 3.0  # at most ~3 dB scalloping anywhere in the FoV
    upa = UniformPlanarArray(4, 4)
    assert len(dft_codebook(upa, 1)) == len(set(zip(dft_codebook(upa, 1).az.round(6), dft_codebook(upa, 1).el.round(6))))
    with pytest.raises(TypeError):
        dft_codebook(object())  # type: ignore[arg-type]
    assert len(grid_codebook(ula, [-10, 0, 10])) == 3


def test_beam_manager_sweeps_periodically_and_tracks_poorly_between_sweeps():
    ula = UniformLinearArray(16, element=Isotropic(0.0))
    cb = dft_codebook(ula, 2, 120.0)
    bm = BeamManager(ula, cb, 1, sweep_period_s=0.5, sweep_duration_s=0.01, meas_noise_db=0.0, rng=rng(0))
    s0 = bm.step(0.0, [0.0], [0.0])
    assert s0.swept[0] and s0.in_sweep[0] and s0.gain_db[0] > 10 * math.log10(16) - 3
    s1 = bm.step(0.1, [math.radians(40.0)], [0.0])  # UAV moved, no new sweep yet
    assert not s1.swept[0] and s1.gain_db[0] < s0.gain_db[0] - 6 and s1.misalignment_deg[0] > 20
    s2 = bm.step(0.6, [math.radians(40.0)], [0.0])  # next sweep recovers the gain
    assert s2.swept[0] and s2.gain_db[0] > 10 * math.log10(16) - 3
    assert bm.sweep_overhead == pytest.approx(0.02)
    bm.reset()


def test_precoders():
    r = rng(1)
    h = (r.normal(size=8) + 1j * r.normal(size=8)) / math.sqrt(2)
    w = mrt_weights(h)
    assert float(np.linalg.norm(w)) == pytest.approx(1.0)
    assert abs(h @ w) == pytest.approx(np.linalg.norm(h))  # MRT achieves the channel norm
    H = (r.normal(size=(3, 8)) + 1j * r.normal(size=(3, 8))) / math.sqrt(2)
    W = zf_weights(H)
    eff = H @ W
    assert np.allclose(eff - np.diag(np.diag(eff)), 0.0, atol=1e-9)  # no inter-user interference
    Wm = mmse_weights(H, 0.01)  # low SNR: close to MRT, so it does not null interference
    assert np.linalg.norm(H @ Wm - np.diag(np.diag(H @ Wm))) > 1e-3
    assert np.allclose(np.linalg.norm(Wm, axis=0), 1.0)
    assert float(pointing_loss_db(math.radians(10), math.radians(10))) == pytest.approx(12.0)
    assert float(pointing_loss_db(math.radians(100), math.radians(1))) == 30.0  # capped


# ---------------------------------------------------------------------------------------- UAV attitude
def test_rotation_helpers():
    R = rpy_to_matrix(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    q = np.array([[1, 0, 0, 0], [math.cos(0.3), 0, 0, math.sin(0.3)]])
    Rq = quat_to_matrices(q)
    assert np.allclose(Rq[0], np.eye(3)) and np.allclose(Rq[1] @ Rq[1].T, np.eye(3))
    # a 90 degree yaw maps the body x axis onto world y
    qyaw = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    assert np.allclose(quat_to_matrices(qyaw) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_direction_in_antenna_frame_follows_attitude_and_mounting():
    az, el = direction_in_antenna_frame([1, 0, 0, 0], [1, 0, 0])
    assert (float(az), float(el)) == (0.0, 0.0)
    # the UAV yaws 90 deg (counter-clockwise): a target on world +y is now straight ahead
    qyaw = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    az, el = direction_in_antenna_frame(qyaw, [0, 1, 0])
    assert float(az) == pytest.approx(0.0, abs=1e-12)
    # straight down is at elevation -90 deg for a level UAV with a forward-mounted antenna
    _, el = direction_in_antenna_frame([1, 0, 0, 0], [0, 0, -1])
    assert float(el) == pytest.approx(-math.pi / 2)
    # an antenna pitched down 90 deg looks at the ground
    az, el = direction_in_antenna_frame([1, 0, 0, 0], [0, 0, -1], (0.0, math.pi / 2, 0.0))
    assert float(az) == pytest.approx(0.0, abs=1e-12) and float(el) == pytest.approx(0.0, abs=1e-12)


def test_uav_antenna_gain_changes_with_tilt():
    ant = UavAntenna.preset("patch_down")
    level = float(ant.gain_dbi([1, 0, 0, 0], [0, 0, -1]))
    assert level == pytest.approx(8.0)
    for tilt in (15.0, 30.0, 60.0):
        q = [math.cos(math.radians(tilt / 2)), 0, math.sin(math.radians(tilt / 2)), 0]
        g = float(ant.gain_dbi(q, [0, 0, -1]))
        assert g == pytest.approx(8.0 - 12.0 * (tilt / 65.0) ** 2, abs=1e-6) or g > 8.0 - 30.0
        assert g < level
    dipole = UavAntenna.preset("dipole")
    gs = dipole.gain_dbi(np.tile([1.0, 0, 0, 0], (3, 1)), np.array([[1.0, 0, 0], [0, 1.0, 0], [0.01, 0, 1.0]]))
    assert gs[0] == pytest.approx(gs[1]) and gs[2] < -10  # omnidirectional in the horizontal plane, null overhead
    with pytest.raises(KeyError):
        UavAntenna.preset("yagi")


# --------------------------------------------------------------------------------------------- MIMO
def test_ergodic_siso_and_mimo_capacity():
    from aevion.wireless.metrics import ergodic_capacity_rayleigh

    c11 = ergodic_capacity(1, 1, 10.0, rng(0), 30000)
    assert c11 == pytest.approx(float(ergodic_capacity_rayleigh(10.0)), abs=0.04)
    c44 = ergodic_capacity(4, 4, 10.0, rng(1), 3000)
    assert c44 > 2.5 * c11  # spatial multiplexing: roughly min(Nr, Nt) times
    # correlation hurts
    R = exponential_correlation(4, 0.9)
    assert ergodic_capacity(4, 4, 10.0, rng(2), 3000, r_rx=R, r_tx=R) < c44


def test_los_dominated_mimo_is_nearly_rank_one():
    ula = UniformLinearArray(4, element=Isotropic(0.0))
    Hlos = geometric_channel(ula, ula, 0.2, 0.0, 0.1, 0.0, 30.0, rng(0))
    Hray = geometric_channel(ula, ula, 0.2, 0.0, 0.1, 0.0, -30.0, rng(0))
    assert condition_number_db(Hlos) > condition_number_db(Hray) + 10
    assert mimo_capacity(Hlos, 10.0) < mimo_capacity(Hray, 10.0)
    assert mimo_capacity(np.eye(2), 10.0) == pytest.approx(2 * math.log2(1 + 10.0 / 2))
    assert water_filling_capacity(Hlos, 10.0) >= mimo_capacity(Hlos, 10.0) - 1e-9  # CSIT cannot hurt
    assert water_filling_capacity(np.zeros((2, 2)), 10.0) == 0.0
    assert water_filling_capacity(np.eye(2), 10.0) == pytest.approx(mimo_capacity(np.eye(2), 10.0))


def test_rayleigh_channel_statistics_and_correlation_matrix():
    H = rayleigh_channel(2, 3, rng(0), size=20000)
    assert H.shape == (20000, 2, 3)
    assert np.mean(np.abs(H) ** 2) == pytest.approx(1.0, abs=0.02)
    R = exponential_correlation(3, 0.7)
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0) and R[0, 2] == pytest.approx(0.49)
    Hc = rayleigh_channel(1, 3, rng(1), r_tx=R, size=40000)[:, 0, :]
    emp = (Hc.conj().T @ Hc / len(Hc)).real
    assert np.allclose(emp, R, atol=0.03)
    with pytest.raises(ValueError):
        exponential_correlation(3, 1.0)


# ------------------------------------------------------------------------------------------- blockage
def test_blockage_probability_boolean_model():
    beta = blockage_rate(1e-3, 1.0)
    assert beta == pytest.approx(2e-3 / math.pi)
    p = blockage_probability([50.0, 100.0], 1e-3, 1.0, h_tx=1.7, h_rx=1.5, h_blocker=1.7)
    assert np.allclose(p, 1 - np.exp(-beta * np.array([50.0, 100.0])))
    # a tall transmitter above the blockers is blocked less
    assert blockage_probability(100.0, 1e-3, 1.0, h_tx=100.0) < blockage_probability(100.0, 1e-3, 1.0, h_tx=3.0)
    assert float(blockage_probability(0.0, 1e-3, 1.0)) == 0.0


def test_blockage_process_reaches_the_stationary_probability_and_is_reproducible():
    bp = BlockageProcess(4000, arrival_rate_hz=0.2, mean_duration_s=1.0, attenuation_db=20.0, rng=rng(0))
    frac = np.mean([bp.step(0.05).mean() / 20.0 for _ in range(400)])
    assert frac == pytest.approx(bp.stationary_blocked_probability, abs=0.01)
    a, b = BlockageProcess(10, rng=rng(3)), BlockageProcess(10, rng=rng(3))
    assert np.array_equal(a.step(0.1), b.step(0.1))
    assert BlockageProcess(5, arrival_rate_hz=0.0, rng=rng(0)).step(1.0).sum() == 0
    with pytest.raises(ValueError):
        BlockageProcess(3, mean_duration_s=0.0)


def test_self_blockage_cone():
    down = self_blockage_loss_db([0, 0, -1.0])
    up = self_blockage_loss_db([0, 0, 1.0])
    edge = self_blockage_loss_db([math.sin(math.radians(45)), 0, -math.cos(math.radians(45))])
    assert float(down) == pytest.approx(15.0) and float(up) == 0.0 and 0.0 < float(edge) < 15.0


# ---------------------------------------------------------------------------------------- delay spread
def test_delay_spread_statistics_follow_tr38901():
    mu, sg = delay_spread_params("uma", False, 3.5)
    assert mu == pytest.approx(-6.28 - 0.204 * math.log10(3.5)) and sg == 0.39
    ds = sample_delay_spread_s("umi", np.zeros(50000, bool), 3.5, rng(0))
    mu, sg = delay_spread_params("umi", False, 3.5)
    assert np.log10(ds).mean() == pytest.approx(mu, abs=0.01) and np.log10(ds).std() == pytest.approx(sg, abs=0.01)
    los = sample_delay_spread_s("uma", np.array([True, False]), 3.5, rng(0))
    assert los.shape == (2,)
    with pytest.raises(KeyError):
        delay_spread_params("moon", True, 1.0)


def test_exponential_pdp_and_coherence_bandwidth():
    tau, p = exponential_pdp(100e-9, 64, span=8.0)
    assert p.sum() == pytest.approx(1.0) and np.all(np.diff(p) < 0)
    assert float(rms_delay_spread(tau, p)) == pytest.approx(100e-9, rel=0.06)
    tau2, p2 = exponential_pdp(100e-9, 16, k_factor_db=10.0)
    assert p2[0] > 0.9 and p2.sum() == pytest.approx(1.0)
    assert float(coherence_bandwidth_hz(1e-6)) == pytest.approx(200e3)
    assert float(coherence_bandwidth_hz(1e-6, 0.9)) == pytest.approx(20e3)
    with pytest.raises(ValueError):
        coherence_bandwidth_hz(1e-6, 0.7)
    f = np.linspace(0, 20e6, 400)
    H = frequency_response(tau, p, f, rng(0))
    assert H.shape == f.shape


# -------------------------------------------------------------------------------------------- mobility
def test_link_kinematics():
    k = link_kinematics([0, 0, 25.0], [100.0, 0, 25.0], [10.0, 0, 0], 3.5e9)
    assert float(k.distance_m) == pytest.approx(100.0)
    assert float(k.radial_speed_mps) == pytest.approx(10.0) and float(k.tangential_speed_mps) == pytest.approx(0.0, abs=1e-9)
    assert float(k.doppler_shift_hz) == pytest.approx(-10.0 * 3.5e9 / 299_792_458.0)  # receding: negative shift
    assert float(k.max_doppler_hz) == pytest.approx(116.75, abs=0.05)
    assert float(k.coherence_time_s) == pytest.approx(0.242 / 116.75, rel=5e-3)
    t = link_kinematics([0, 0, 0.0], [100.0, 0, 0], [0.0, 20.0, 0], 2e9)  # purely tangential flight
    assert float(t.doppler_shift_hz) == pytest.approx(0.0, abs=1e-9) and float(t.angular_rate_deg_s) == pytest.approx(math.degrees(0.2))
    batch = link_kinematics(np.zeros((1, 3, 3)) + [0, 0, 20], np.array([[[10, 0, 0], [0, 20, 0], [30, 30, 0]]]), np.zeros((1, 3, 3)) + [1, 0, 0], 2e9)
    assert batch.distance_m.shape == (1, 3)
