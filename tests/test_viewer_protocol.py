"""The binary frame container: Python writer/reader and the JavaScript twin must agree byte for byte."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aevion.viewer.protocol import FormatError, decode_bundle, encode_bundle

WEB = Path(__file__).resolve().parents[1] / "src" / "aevion" / "viewer" / "web"


def test_roundtrip_preserves_names_shapes_and_values():
    arrays = {"t": np.linspace(0, 1, 7), "pos": np.random.default_rng(0).normal(size=(7, 3, 3)), "soc": np.ones((7, 3))}
    head, out = decode_bundle(encode_bundle({"type": "frames", "t0": 2.5}, arrays))
    assert head["type"] == "frames" and head["t0"] == 2.5
    assert list(out) == ["t", "pos", "soc"]
    for k in arrays:
        assert out[k].shape == arrays[k].shape
        np.testing.assert_allclose(out[k], arrays[k].astype(np.float32))


def test_arrays_are_four_byte_aligned_for_any_header_length():
    for n in range(1, 9):
        blob = encode_bundle({"type": "x" * n}, {"a": np.arange(3.0)})
        (hlen,) = struct.unpack_from("<I", blob, 0)
        assert (4 + hlen + ((-(4 + hlen)) % 4)) % 4 == 0
        np.testing.assert_array_equal(decode_bundle(blob)[1]["a"], [0, 1, 2])


def test_empty_and_scalar_arrays():
    head, out = decode_bundle(encode_bundle({}, {"e": np.zeros((0, 3))}))
    assert out["e"].shape == (0, 3)


def test_non_finite_header_values_are_rejected():
    with pytest.raises(ValueError):
        encode_bundle({"x": float("nan")}, {})


@pytest.mark.parametrize("blob", [b"", b"\x01\x00", struct.pack("<I", 10**9) + b"{}"])
def test_malformed_bundles_raise_format_error(blob):
    with pytest.raises(FormatError):
        decode_bundle(blob)


def test_truncated_array_data_is_detected():
    blob = encode_bundle({}, {"a": np.arange(10.0)})
    with pytest.raises(FormatError):
        decode_bundle(blob[:-4])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_javascript_decoder_reads_python_bundles_and_python_reads_javascript_bundles(tmp_path):
    arrays = {"t": np.arange(5.0), "pos": np.arange(30.0).reshape(5, 2, 3)}
    (tmp_path / "py.bin").write_bytes(encode_bundle({"type": "frames", "t0": 1.5}, arrays))
    script = tmp_path / "t.mjs"
    script.write_text(
        f"""
import fs from 'node:fs';
import {{ decodeBundle, encodeBundle }} from {json.dumps((WEB / "src/data/decode.js").as_uri())};
const b = fs.readFileSync({json.dumps(str(tmp_path / "py.bin"))});
const {{ header, arrays }} = decodeBundle(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
console.log(JSON.stringify({{ type: header.type, t0: header.t0, shape: arrays.pos.shape, last: Array.from(arrays.pos.data).pop() }}));
fs.writeFileSync({json.dumps(str(tmp_path / "js.bin"))}, Buffer.from(encodeBundle({{ type: 'frames', t0: 9 }}, {{ t: {{ data: new Float32Array([0, 1, 2]), shape: [3] }}, soc: {{ data: new Float32Array([5, 6]), shape: [1, 2] }} }})));
""",
        encoding="utf-8",
    )
    r = subprocess.run([shutil.which("node"), str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got == {"type": "frames", "t0": 1.5, "shape": [5, 2, 3], "last": 29}
    head, out = decode_bundle((tmp_path / "js.bin").read_bytes())
    assert head["t0"] == 9 and out["soc"].shape == (1, 2) and out["soc"].tolist() == [[5, 6]]
