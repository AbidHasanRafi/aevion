"""The single-file bundler and the packaged web app."""

import hashlib
import json
import re
import shutil
import subprocess

import pytest
from viewer_support import synthetic_payload

from aevion.viewer.export import VENDOR_SPECIFIERS, _rewrite_imports, bundle_modules, export_html, render_html, web_root

WEB = web_root()
IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"]+)\1""")


def test_every_import_in_the_app_resolves_to_a_bundled_module():
    mods = bundle_modules()
    for spec, src in mods.items():
        if not spec.startswith("@aevion/"):
            continue
        for m in IMPORT.finditer(src):
            target = m.group(2)
            assert not target.startswith("."), f"{spec}: relative import {target!r} survived the rewrite"
            assert target in mods, f"{spec} imports {target!r}, which is not bundled"


def test_all_source_files_are_bundled_and_vendor_files_present():
    mods = bundle_modules()
    for p in (WEB / "src").rglob("*.js"):
        assert f"@aevion/{p.relative_to(WEB / 'src').as_posix()}" in mods
    assert set(VENDOR_SPECIFIERS) <= set(mods)


def test_rewrite_resolves_parent_and_sibling_imports():
    src = "import a from './x.js';\nimport { b } from '../data/y.js';\nconst m = await import('./z.js');\nimport 'three';"
    out = _rewrite_imports(src, "ui/panel.js")
    assert "'@aevion/ui/x.js'" in out and "'@aevion/data/y.js'" in out and "'@aevion/ui/z.js'" in out and "'three'" in out
    with pytest.raises(ValueError):
        _rewrite_imports("import x from '../../../etc.js';", "a.js")


def test_html_has_no_unresolved_placeholders_and_embeds_valid_json(tmp_path):
    payload = synthetic_payload(2, 20)
    page = render_html(payload)
    assert "<!--aevion:dev-->" not in page and "/*AEVION_BOOT*/" not in page and 'src="src/app.js"' not in page
    body = page.split('id="aevion-payload">')[1].split("</script>")[0]
    assert json.loads(body)["meta"]["n"] == 2
    mods = json.loads(page.split('id="aevion-modules">')[1].split("</script>")[0])
    assert "@aevion/app.js" in mods and "three" in mods


def test_hostile_titles_cannot_break_out_of_the_page():
    payload = synthetic_payload(1, 10)
    payload["meta"]["title"] = "</script><script>alert(1)</script><!--"
    page = render_html(payload)
    assert "<script>alert(1)" not in page
    assert page.count("</script>") == page.count("<script")  # every script element is closed by its own tag, none by the title
    assert "<title>&lt;/script&gt;" in page or "<title>&lt;/script>" in page


def test_export_is_deterministic(tmp_path):
    a = render_html(synthetic_payload(2, 30, seed=1))
    assert a == render_html(synthetic_payload(2, 30, seed=1))


def test_export_embeds_a_vehicle_model_and_rejects_other_files(tmp_path):
    from viewer_support import sim_archive

    a = sim_archive(duration=1.0)
    glb = tmp_path / "m.glb"
    glb.write_bytes(b"glTF" + b"\0" * 12)
    page = export_html(a, tmp_path / "r.html", vehicle_model=glb).read_text(encoding="utf-8")
    assert "data:model/gltf-binary;base64," in page
    bad = tmp_path / "m.obj"
    bad.write_text("v 0 0 0")
    with pytest.raises(ValueError):
        export_html(a, tmp_path / "x.html", vehicle_model=bad)


def test_export_option_theme_and_legacy_offline_flag(tmp_path):
    from viewer_support import sim_archive

    a = sim_archive(duration=1.0)
    page = export_html(a, tmp_path / "r.html", offline=False, theme="print").read_text(encoding="utf-8")
    assert '{"theme": "print"}' in page


def test_vendored_three_matches_its_manifest():
    manifest = json.loads((WEB / "vendor" / "three" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.160.0"
    for name, info in manifest["files"].items():
        data = (WEB / "vendor" / "three" / name).read_bytes()
        want = info.get("sha256_patched", info["sha256"])
        assert hashlib.sha256(data).hexdigest() == want, f"{name} differs from the manifest (edited or corrupted)"
    assert (WEB / "vendor" / "three" / "LICENSE").exists()


def test_web_app_has_no_network_references_other_than_local_ones():
    for p in list((WEB / "src").rglob("*.js")) + [WEB / "index.html", WEB / "css" / "aevion.css"]:
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"https?://[^\s'\"`)]+", text):
            assert m.group(0).startswith(("http://www.w3.org",)), f"{p.name}: external URL {m.group(0)}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_node_unit_tests_pass():
    r = subprocess.run(
        [shutil.which("node"), "--test", *map(str, sorted((WEB / "tests").glob("*.test.mjs")))], capture_output=True, text=True, cwd=WEB
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
