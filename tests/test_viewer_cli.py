"""Command-line entry points of the viewer (no browser needed)."""

import pytest
from viewer_support import sim_archive

from aevion.cli import build_parser, main


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    a, b = sim_archive(duration=1.5), sim_archive(duration=1.5, wind={"type": "ConstantWind", "velocity": [2, 0, 0]})
    return a.save(d / "a"), b.save(d / "b")


def test_replay_writes_a_single_html_file(archives, tmp_path, capsys):
    out = tmp_path / "r.html"
    main(["replay", str(archives[0]), "--out", str(out), "--theme", "print"])
    assert out.stat().st_size > 500_000 and "wrote" in capsys.readouterr().out
    assert '"theme": "print"' in out.read_text(encoding="utf-8")


def test_replay_compare_uses_labels(archives, tmp_path):
    out = tmp_path / "c.html"
    main(["replay", str(archives[0]), "--compare", str(archives[1]), "--label", "calm", "--label", "windy", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "calm vs windy" in text and '"runs"' in text


def test_replay_label_count_must_match(archives, tmp_path):
    with pytest.raises(SystemExit, match="--label"):
        main(["replay", str(archives[0]), "--compare", str(archives[1]), "--label", "only-one"])


@pytest.mark.parametrize("size", ["big", "10x10", "99999x100", "1280"])
def test_bad_sizes_are_rejected_before_any_browser_starts(archives, size):
    with pytest.raises(SystemExit, match="--size"):
        main(["replay", str(archives[0]), "--screenshot", "x.png", "--size", size])


def test_edit_writes_the_editor_page(tmp_path, capsys):
    out = tmp_path / "ed.html"
    main(["edit", "--out", str(out)])
    assert out.exists() and "aevion run scene.json" in capsys.readouterr().out


def test_parser_knows_the_new_commands():
    p = build_parser()
    for argv in (
        ["live", "x.yaml", "--speed", "0", "--no-wait"],
        ["edit", "--open"],
        ["replay", "a", "--serve", "--port", "0", "--video", "v.mp4", "--sweep", "90"],
    ):
        assert p.parse_args(argv).fn
