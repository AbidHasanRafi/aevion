"""Diagnostic figures, their numerics, journal templates and the methods appendix - against closed forms, scikit-learn and golden images."""

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from aevion.analysis import diagnostics as D
from aevion.analysis import figures_diag as FD
from aevion.analysis import style as S
from aevion.analysis.appendix import METRIC_GLOSSARY, methods_appendix
from aevion.analysis.templates import available_venues, figure_latex, latex_escape, venue_summary
from aevion.experiments import ConfigExperiment

LAG = ([1.0], [1.0, 1.0])  # 1 / (s + 1)
L3 = ([2.0], [1.0, 3.0, 2.0, 0.0])  # 2 / (s (s + 1) (s + 2))


# ------------------------------------------------------------------------------------------------ frequency response and margins
def test_a_first_order_lag_is_minus_three_decibels_and_forty_five_degrees_at_its_corner():
    H = D.freqresp(LAG, np.array([0.01, 1.0, 100.0]))
    assert 20 * np.log10(abs(H[1])) == pytest.approx(-3.0103, abs=1e-3) and np.degrees(np.angle(H[1])) == pytest.approx(-45.0)
    assert abs(H[0]) == pytest.approx(1.0, abs=1e-4) and abs(H[2]) == pytest.approx(0.01, rel=1e-3)


def test_transfer_function_state_space_and_scipy_objects_describe_the_same_system():
    from scipy import signal

    A, B, C, Dm = [[0, 1, 0], [0, 0, 1], [0, -2, -3]], [[0], [0], [1]], [[2, 0, 0]], [[0]]  # controllable form of 2 / (s^3 + 3 s^2 + 2 s)
    w = np.logspace(-2, 2, 25)
    ref = D.freqresp(L3, w)
    assert np.allclose(D.freqresp((A, B, C, Dm), w), ref, rtol=1e-9)
    assert np.allclose(D.freqresp(signal.TransferFunction(*L3), w), ref)
    p, z = D.poles_zeros(L3)
    assert np.allclose(sorted(p.real), [-2, -1, 0]) and len(z) == 0
    with pytest.raises(ValueError):
        D.tf_from([1, 2, 3])


def test_stability_margins_match_the_textbook_values():
    m = D.margins(L3)  # K = 2 on a unity-feedback loop 1 / (s (s+1) (s+2)): the loop is unstable for K > 6
    assert m["gain_margin"] == pytest.approx(3.0, rel=1e-6) and m["w_pc"] == pytest.approx(math.sqrt(2), rel=1e-6)
    ph = -90 - math.degrees(math.atan(m["w_gc"])) - math.degrees(math.atan(m["w_gc"] / 2))  # phase of L at the gain crossover
    assert (
        m["phase_margin_deg"] == pytest.approx(180 + ph, abs=1e-6)
        and 25 < m["phase_margin_deg"] < 40
        and abs(D.freqresp(L3, np.array([m["w_gc"]]))[0]) == pytest.approx(1.0)
    )
    integ = D.margins(([5.0], [1.0, 0.0]))  # L = 5 / s: crossover at 5 rad/s with a 90 degree margin, never a -180 degree crossing
    assert integ["w_gc"] == pytest.approx(5.0) and integ["phase_margin_deg"] == pytest.approx(90.0) and integ["gain_margin"] == math.inf
    di = D.margins(([1.0], [1.0, 1.0, 0.0]))  # 1 / (s (s + 1)): wc = 0.786, PM = 90 - atan(wc) = 51.8 degrees
    wc = math.sqrt((-1 + math.sqrt(5)) / 2)
    assert di["w_gc"] == pytest.approx(wc, rel=1e-6) and di["phase_margin_deg"] == pytest.approx(90 - math.degrees(math.atan(wc)), abs=1e-6)
    lag = D.margins(LAG)  # |L| < 1 everywhere: no gain crossover
    assert math.isnan(lag["phase_margin_deg"]) and lag["gain_margin"] == math.inf
    unstable = D.margins(([12.0], [1.0, 3.0, 2.0, 0.0]))  # K = 12 > 6: the loop is closed-loop unstable and the margins say so
    assert unstable["gain_margin"] < 1.0 and unstable["phase_margin_deg"] < 0


# ------------------------------------------------------------------------------------------------ ROC and KDE
def test_roc_curves_have_the_right_extremes_and_agree_with_scikit_learn():
    y = np.r_[np.ones(50), np.zeros(50)].astype(bool)
    assert D.auc(y, np.r_[np.arange(50) + 100.0, np.arange(50)]) == 1.0 and D.auc(y, -np.r_[np.arange(50) + 100.0, np.arange(50)]) == 0.0
    fpr, tpr, thr = D.roc_curve(y, np.r_[np.ones(50), np.zeros(50)])
    assert (
        fpr[0] == 0
        and tpr[0] == 0
        and fpr[-1] == 1
        and tpr[-1] == 1
        and np.all(np.diff(fpr) >= 0)
        and np.all(np.diff(tpr) >= 0)
        and thr[0] == math.inf
    )
    assert D.auc(y, np.ones(100)) == pytest.approx(0.5)  # constant scores carry no information
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    for _ in range(5):
        s = np.round(rng.normal(size=300) + 0.8 * rng.integers(0, 2, 300), 1)  # rounding creates ties
        lab = rng.integers(0, 2, 300).astype(bool)
        assert D.auc(lab, s) == pytest.approx(metrics.roc_auc_score(lab, s), abs=1e-12)
        f, t, _ = D.roc_curve(lab, s)
        sf, st, _ = metrics.roc_curve(lab, s, drop_intermediate=False)
        assert np.allclose(f, sf) and np.allclose(t, st)
    # equals the Mann-Whitney probability P(pos > neg) + 0.5 P(tie)
    pos, neg = s[lab], s[~lab]
    prob = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    assert D.auc(lab, s) == pytest.approx(prob, abs=1e-12)


def test_the_auc_of_two_gaussians_is_phi_of_d_prime_over_root_two_and_the_interval_covers_it():
    rng = np.random.default_rng(1)
    n, d = 3000, 1.2
    y = np.r_[np.ones(n), np.zeros(n)].astype(bool)
    s = np.r_[rng.normal(d, 1, n), rng.normal(0, 1, n)]
    a, lo, hi = D.auc_ci(y, s, n_boot=200)
    assert a == pytest.approx(norm.cdf(d / math.sqrt(2)), abs=0.01) and lo < norm.cdf(d / math.sqrt(2)) < hi and hi - lo < 0.04
    for bad in ((np.ones(5, bool), np.arange(5.0)), (np.array([True, False]), np.array([1.0, np.nan])), (np.array([True]), np.array([1.0, 2.0]))):
        with pytest.raises(ValueError):
            D.roc_curve(*bad)


def test_the_kernel_density_estimate_integrates_to_one_and_peaks_where_the_data_do():
    x = np.random.default_rng(2).normal(3.0, 0.5, 2000)
    g, dens = D.kde_curve(x)
    assert np.trapezoid(dens, g) == pytest.approx(1.0, abs=5e-3) and g[np.argmax(dens)] == pytest.approx(3.0, abs=0.1)
    assert dens.max() == pytest.approx(norm.pdf(0, scale=0.5), rel=0.1)
    assert D.bandwidth(x, 0.2) == 0.2 and D.bandwidth(x, "silverman") > 0
    assert D.kde_curve(x, bw=0.05)[1].max() > dens.max()  # a smaller bandwidth gives a spikier estimate
    with pytest.raises(ValueError):
        D.kde_curve([1.0])
    with pytest.raises(ValueError):
        D.bandwidth(x, "magic")


# ------------------------------------------------------------------------------------------------ figures
def lines_of(ax):
    return [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 20]


def test_the_bode_figure_plots_the_analytic_response_and_marks_the_margins():
    fig = FD.fig_bode(L3)
    mag, ph = fig.axes
    w, y = lines_of(mag)[0].get_data()
    assert np.allclose(y, 20 * np.log10(abs(D.freqresp(L3, w)))) and np.allclose(
        lines_of(ph)[0].get_ydata(), np.degrees(np.unwrap(np.angle(D.freqresp(L3, w))))
    )
    texts = [t.get_text() for ax in fig.axes for t in ax.get_legend().get_texts()]
    assert any(t.startswith("PM = ") for t in texts) and any(t.startswith("GM = 9.5 dB") for t in texts)  # 20 log10(3) = 9.54 dB
    assert mag.get_ylabel().startswith("Magnitude") and ph.get_xlabel().startswith("Frequency")
    multi = FD.fig_bode({"a": LAG, "b": ([1.0], [1.0, 2.0])}, show_margins=False)
    assert len(lines_of(multi.axes[0])) == 2
    plt.close("all")


def test_the_pole_zero_map_flags_unstable_poles_and_draws_the_right_boundary():
    fig = FD.fig_pzmap(([1.0, -1.0], [1.0, -1.0, 2.0]))  # pole pair at 0.5 +/- 1.32i: unstable; a zero at 1
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "unstable pole" in labels and "zeros" in labels
    assert ax.get_xlim()[1] > 0.5 and ax.get_xlim()[0] < 0
    disc = FD.fig_pzmap(([1.0], [1.0, -1.2, 0.35]), discrete=True)  # poles at 0.5 and 0.7: inside the unit circle
    d_ax = disc.axes[0]
    circle = [ln for ln in d_ax.get_lines() if len(ln.get_xdata()) == 200][0]
    assert np.allclose(np.hypot(circle.get_xdata(), circle.get_ydata()), 1.0) and "unstable pole" not in [ln.get_label() for ln in d_ax.get_lines()]
    plt.close("all")


def test_the_allan_figure_draws_only_the_asymptotes_that_were_given():
    taus = np.geomspace(0.1, 100, 40)
    fig = FD.fig_allan(taus, 0.3 / np.sqrt(taus), {"N": 0.3, "B": 0.01})
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(texts) == 3 and "measured" in texts and not any("K" in t for t in texts)
    n_line = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"][0]
    assert np.allclose(n_line.get_ydata() * np.sqrt(n_line.get_xdata()), 0.3)  # the N / sqrt(tau) asymptote
    plt.close("all")


def test_the_nees_figure_reports_the_fraction_inside_the_chi_square_band():
    from aevion.estimation.consistency import nees_interval

    rng = np.random.default_rng(0)
    runs = rng.chisquare(3, (200, 80))  # a consistent filter: chi-square with 3 degrees of freedom
    fig = FD.fig_nees(np.arange(80) * 0.1, runs, 3)
    ax = fig.axes[0]
    lo, hi = nees_interval(3, 200)
    band = ax.patches[0]
    assert band.get_y() == pytest.approx(lo) and band.get_y() + band.get_height() == pytest.approx(hi)
    label = [t.get_text() for t in ax.get_legend().get_texts() if "inside" in t.get_text()][0]
    assert int(label.split("(")[1].split("%")[0]) >= 90
    bad = FD.fig_nees(np.arange(80) * 0.1, 3.0 * runs.mean(axis=0), 3, n_runs=200)  # an over-confident filter: NEES three times too large
    assert int([t.get_text() for t in bad.axes[0].get_legend().get_texts() if "inside" in t.get_text()][0].split("(")[1].split("%")[0]) == 0
    plt.close("all")


def test_the_roc_figure_lists_the_auc_of_each_curve():
    rng = np.random.default_rng(4)
    y = np.r_[np.ones(200), np.zeros(200)].astype(bool)
    good, poor = np.r_[rng.normal(2, 1, 200), rng.normal(0, 1, 200)], np.r_[rng.normal(0.3, 1, 200), rng.normal(0, 1, 200)]
    fig = FD.fig_roc({"good": (y, good), "poor": (y, poor)})
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts[0].startswith("good (AUC 0.9") and texts[1].startswith("poor (AUC 0.5") and texts[-1] == "chance"
    assert ax.get_xlim() == (0, 1) and ax.get_aspect() == 1.0
    assert "[" in [t.get_text() for t in FD.fig_roc({"g": (y, good)}, ci=True).axes[0].get_legend().get_texts()][0]
    plt.close("all")


def test_the_ridgeline_figure_has_one_labelled_ridge_per_group_in_order():
    rng = np.random.default_rng(5)
    fig = FD.fig_ridgeline(
        {"calm": rng.normal(0.3, 0.05, 80), "windy": rng.normal(0.6, 0.1, 80), "gusty": rng.normal(1.0, 0.2, 80)}, xlabel="RMSE (m)"
    )
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["gusty", "windy", "calm"] and ax.get_xlabel() == "RMSE (m)"  # first group on top
    assert len(ax.collections) == 3
    with pytest.raises(ValueError):
        FD.fig_ridgeline({})
    plt.close("all")


def test_the_step_response_figure_quotes_the_step_metrics():
    t = np.linspace(0, 8, 400)
    y = 1 - np.exp(-t)  # first order: rise time 2.2 tau, no overshoot
    fig = FD.fig_step_response(t, y, 1.0)
    text = " ".join(tx.get_text() for tx in fig.axes[0].get_legend().get_texts())
    assert "rise 2.19 s" in text and "overshoot 0.0 %" in text
    plt.close("all")


@pytest.mark.parametrize("venue", sorted(S.VENUES))
def test_figures_follow_the_venue_width_and_typeface(venue):
    info = S.VENUES[venue]
    fig = FD.fig_roc({"a": (np.r_[np.ones(20), np.zeros(20)].astype(bool), np.arange(40.0))}, venue=venue)
    assert fig.get_size_inches()[0] == pytest.approx(info["single"])
    fam = fig.axes[0].xaxis.label.get_fontfamily()[0]
    assert fam == info["family"]
    plt.close(fig)
    wide = FD.fig_bode(L3, columns="double", venue=venue)
    assert wide.get_size_inches()[0] == pytest.approx(info["double"])
    plt.close(wide)


def test_unknown_venues_and_widths_are_refused():
    with pytest.raises(ValueError, match="unknown venue"):
        S.figsize("single", venue="nature")
    with pytest.raises(ValueError, match="one_half"):
        S.figsize("one_half", venue="ieee")
    assert S.figsize("one_half", venue="elsevier")[0] == pytest.approx(5.51)


# ------------------------------------------------------------------------------------------------ templates
def test_venue_summaries_and_latex_wrappers():
    assert available_venues() == sorted(S.VENUES) and "elsevier" in available_venues()
    s = venue_summary("ieee")
    assert s["widths_in"] == {"single": 3.5, "double": 7.16} and s["widths_mm"]["single"] == pytest.approx(88.9) and s["family"] == "serif"
    e = venue_summary("elsevier")
    assert (
        e["family"] == "sans-serif"
        and e["widths_mm"]["single"] == pytest.approx(90.0, abs=0.2)
        and e["widths_mm"]["double"] == pytest.approx(190.0, abs=0.2)
    )
    one = figure_latex("fig/bode", "Open-loop response.", "fig:bode", "ieee")
    assert (
        one.splitlines()[0] == "\\begin{figure}[t]"
        and "\\includegraphics[width=3.50in]{fig/bode}" in one
        and one.endswith("\\end{figure}")
        and "\\Description" not in one
    )
    two = figure_latex("fig/wide", "Wide.", "fig:wide", "ieee", "double")
    assert two.startswith("\\begin{figure*}[!t]") and "width=7.16in" in two and two.endswith("\\end{figure*}")
    acm = figure_latex("f", "A 50% gain & more", "fig:a", "acm")
    assert "\\Description{A 50\\% gain \\& more}" in acm  # the alternative text is escaped
    assert "[htbp]" in figure_latex("f", "c", "l", "elsevier") and "width=5.51in" in figure_latex("f", "c", "l", "elsevier", "one_half")
    assert "\\begin{figure}[ht]" in figure_latex("f", "c", "l", "springer") and latex_escape("a_b #1 {x}") == "a\\_b \\#1 \\{x\\}"
    for bad in (dict(label="has space"), dict(label=""), dict(label="a{b}"), dict(columns="triple")):
        with pytest.raises(ValueError):
            figure_latex("f", "c", **{"label": "l", **bad})
    with pytest.raises(ValueError, match="unknown venue"):
        venue_summary("nature")


# ------------------------------------------------------------------------------------------------ methods appendix
def small_experiment():
    base = {
        "duration": 3.0,
        "seed": 1,
        "trajectory": {"type": "circle", "params": {"center": [0, 0, 3.0], "radius": 2.0, "omega": 0.7}},
        "wind": {"type": "constant", "velocity": [2.0, 0.0, 0.0]},
        "sensors": ["imu"],
    }
    return ConfigExperiment(
        "m",
        base,
        {"geometric": {"controller": "geometric"}, "lqr": {"controller": "lqr"}},
        {"calm": {}, "windy": {"wind": {"type": "constant", "velocity": [4.0, 0.0, 0.0]}}},
        seeds=[0, 1, 2],
        metric_params={"skip": 0.5},
    )


def test_the_appendix_states_the_numbers_that_were_actually_run():
    exp = small_experiment()
    res = exp.run()
    app = methods_appendix(result=res, experiment=exp, metrics=["rmse_pos", "energy_wh", "not_a_metric"])
    md = app.markdown()
    assert [t for t, _ in app.sections] == ["Simulation", "Experimental design", "Metrics", "Statistical analysis", "Reproducibility"]
    assert "`rk4`" in md and "5 ms" in md and "3 s per run" in md and "generic_1kg" in md and "circle" in md and "constant" in md
    assert (
        "2 method(s)" in md
        and "`geometric`" in md
        and "`lqr`" in md
        and "2 condition(s)" in md
        and "3 random seed(s)" in md
        and "12 simulation runs" in md
    )
    assert "`rmse_pos`: root-mean-square 3-D position error" in md and "not_a_metric" not in md
    assert "Holm" in md and "Welch" in md and "Mann" in md and "Cliff" in md
    assert "Aevion " in md and "bit for bit" in md
    assert (
        methods_appendix(result=res, experiment=exp, metrics=["rmse_pos"]).markdown()
        == methods_appendix(result=res, experiment=exp, metrics=["rmse_pos"]).markdown()
    )  # deterministic
    only = methods_appendix(
        config={"duration": 2.0, "sensors": [], "faults": [{"type": "payload_change", "t_start": 1.0, "params": {"mass": 0.3}}]}
    ).markdown()
    assert "Experimental design" not in only and "payload_change" in only and "from t = 1 s" in only


def test_the_latex_form_escapes_special_characters_and_uses_lists():
    exp = small_experiment()
    app = methods_appendix(experiment=exp, metrics=["rmse_pos", "energy_wh"], config=exp.base)
    tex = app.latex()
    assert "\\subsection*{Simulation}" in tex and "\\begin{itemize}" in tex and "\\item" in tex and "\\end{itemize}" in tex
    assert "rmse\\_pos" in tex and "95 \\%" in tex and "_pos" not in tex.replace("\\_pos", "")
    assert app.text("latex") == tex and app.text() == app.markdown()
    with pytest.raises(ValueError):
        app.text("rtf")
    assert all(isinstance(v, str) and v for v in METRIC_GLOSSARY.values())


# ------------------------------------------------------------------------------------------------ golden images
GOLD = Path(__file__).parent / "golden" / "figures"
META = GOLD / "meta.json"
DPI = 100


def _cases():
    rng = np.random.default_rng(7)
    y = np.r_[np.ones(150), np.zeros(150)].astype(bool)
    taus = np.geomspace(0.1, 100, 40)
    return {
        "p6_bode": lambda: FD.fig_bode(L3),
        "p6_pzmap": lambda: FD.fig_pzmap({"closed loop": ([2.0], [1.0, 3.0, 2.0, 2.0])}),
        "p6_allan": lambda: FD.fig_allan(
            taus, np.sqrt((0.3**2) / taus + (0.01 * 0.664) ** 2 + (0.02**2) * taus / 3), {"N": 0.3, "B": 0.01, "K": 0.02}
        ),
        "p6_nees": lambda: FD.fig_nees(np.arange(80) * 0.1, np.random.default_rng(1).chisquare(3, (50, 80)), 3),
        "p6_roc": lambda: FD.fig_roc(
            {
                "detector A": (y, np.r_[rng.normal(1.8, 1, 150), rng.normal(0, 1, 150)]),
                "detector B": (y, np.r_[rng.normal(0.8, 1, 150), rng.normal(0, 1, 150)]),
            }
        ),
        "p6_ridgeline": lambda: FD.fig_ridgeline(
            {"calm": rng.normal(0.3, 0.05, 80), "windy": rng.normal(0.6, 0.1, 80), "gusty": rng.normal(1.0, 0.2, 80)}, xlabel="RMSE (m)"
        ),
        "p6_bode_elsevier": lambda: FD.fig_bode(L3, venue="elsevier"),
    }


@pytest.fixture(scope="module", autouse=True)
def _deterministic_fonts():
    old_serif, old_sans = S.SERIF_PREFERENCE, S.SANS_PREFERENCE
    S.SERIF_PREFERENCE, S.SANS_PREFERENCE = ["STIXGeneral"], ["DejaVu Sans"]
    yield
    S.SERIF_PREFERENCE, S.SANS_PREFERENCE = old_serif, old_sans


def _signature():
    return {"matplotlib": matplotlib.__version__.rsplit(".", 1)[0], "numpy_major": int(np.__version__.split(".")[0])}


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(_cases()))
def test_diagnostic_figure_matches_golden(name, tmp_path, request):
    golden = GOLD / f"{name}.png"
    fig = _cases()[name]()
    if request.config.getoption("--update-goldens"):
        GOLD.mkdir(parents=True, exist_ok=True)
        fig.savefig(golden, dpi=DPI)
        plt.close(fig)
        return
    if not golden.exists() or not META.exists():
        plt.close(fig)
        pytest.skip("no golden images yet (python tools/update_goldens.py)")
    if json.loads(META.read_text(encoding="utf-8")) != _signature():
        plt.close(fig)
        pytest.skip("goldens were rendered with a different Matplotlib/NumPy")
    actual = tmp_path / f"{name}.png"
    fig.savefig(actual, dpi=DPI)
    plt.close(fig)
    err = compare_images(str(golden), str(actual), 3.0, in_decorator=False)
    assert err is None, f"{name} differs from its golden image:\n{err}"
