"""Publication figures: structural checks (always) and pixel goldens (when the rendering stack matches).

Goldens are rendered with the font bundled in Matplotlib (STIX) so they do not depend on which Times-like font a
machine has installed. Regenerate with ``python tools/update_goldens.py`` after an intentional visual change.
"""

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from aevion.analysis import figures as F
from aevion.analysis import style as S
from aevion.engine import Simulation

GOLD = Path(__file__).parent / "golden" / "figures"
META = GOLD / "meta.json"
DPI = 100
TOL_RMS = 3.0

CASES = {
    "trajectory_3d": lambda a: F.fig_trajectory_3d(a),
    "topdown": lambda a: F.fig_topdown(a),
    "tracking": lambda a: F.fig_tracking(a),
    "attitude": lambda a: F.fig_attitude(a),
    "rotors": lambda a: F.fig_rotors(a),
    "energy": lambda a: F.fig_energy(a),
    "error_distribution": lambda a: F.fig_error_distribution(a),
    "psd": lambda a: F.fig_psd(a),
    "overview": lambda a: F.fig_overview(a),
}

CONFIG = {
    "schema_version": 2,
    "name": "golden-figures",
    "seed": 7,
    "duration": 6.0,
    "vehicle": {"model": "generic_1kg"},
    "controller": {"type": "geometric"},
    "trajectory": {"type": "figure8", "params": {"center": [0, 0, 2.0], "size": 2.0, "period": 6.0}},
    "wind": {"type": "constant", "params": {"velocity": [0.8, 0.3, 0.0]}},
    "sensors": ["imu"],
}


@pytest.fixture(scope="module", autouse=True)
def deterministic_font():
    old = S.SERIF_PREFERENCE
    S.SERIF_PREFERENCE = ["STIXGeneral"]
    yield
    S.SERIF_PREFERENCE = old


@pytest.fixture(scope="module")
def archive():
    return Simulation(CONFIG).run()


def render(name, archive, path):
    fig = CASES[name](archive)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def stack_signature():
    import numpy as np

    return {"matplotlib": matplotlib.__version__.rsplit(".", 1)[0], "numpy_major": int(np.__version__.split(".")[0])}


# ---------------------------------------------------------------------------------------- structure
@pytest.mark.parametrize("name", sorted(CASES))
def test_figure_is_well_formed(name, archive):
    fig = CASES[name](archive)
    try:
        assert fig.get_axes(), "no axes"
        width, height = fig.get_size_inches()
        assert 3.2 <= width <= 7.3 and 1.5 <= height <= 9.5, f"{width}x{height} in is not a journal column size"
        for ax in fig.get_axes():
            if ax.get_label() == "<colorbar>" or not ax.axison:
                continue
            assert ax.get_xlabel() or ax.get_ylabel() or ax.get_title() or ax.name == "3d", f"unlabelled axes in {name}"
        fonts = {t.get_fontfamily()[0] for ax in fig.get_axes() for t in [ax.xaxis.label, ax.yaxis.label] if t.get_text()}
        assert fonts <= {"serif"}, fonts
    finally:
        plt.close(fig)


def test_figures_are_deterministic(archive, tmp_path):
    render("tracking", archive, tmp_path / "a.png")
    render("tracking", archive, tmp_path / "b.png")
    assert compare_images(str(tmp_path / "a.png"), str(tmp_path / "b.png"), 0.0) is None


def test_stix_is_available_for_the_golden_font():
    from matplotlib import font_manager

    assert "STIXGeneral" in {f.name for f in font_manager.fontManager.ttflist}


# ---------------------------------------------------------------------------------------- pixel goldens
@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(CASES))
def test_figure_matches_golden(name, archive, tmp_path, request):
    golden = GOLD / f"{name}.png"
    if request.config.getoption("--update-goldens"):
        GOLD.mkdir(parents=True, exist_ok=True)
        render(name, archive, golden)
        META.write_text(json.dumps(stack_signature(), indent=1), encoding="utf-8")
        return
    if not golden.exists() or not META.exists():
        pytest.skip("no golden images yet (python tools/update_goldens.py)")
    if json.loads(META.read_text(encoding="utf-8")) != stack_signature():
        pytest.skip(f"goldens were rendered with {json.loads(META.read_text(encoding='utf-8'))}, this is {stack_signature()}")
    actual = tmp_path / f"{name}.png"
    render(name, archive, actual)
    err = compare_images(str(golden), str(actual), TOL_RMS, in_decorator=False)
    assert err is None, f"{name} differs from its golden image:\n{err}\n(intentional? run python tools/update_goldens.py)"
