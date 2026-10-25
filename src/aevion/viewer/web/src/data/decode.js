// Binary container shared by the HTTP frame API and the live WebSocket stream.
//
//   [uint32 little-endian: header length H][H bytes of UTF-8 JSON][padding to a multiple of 4][float32 arrays back to back]
//
// The JSON header is  {"type": ..., "arrays": [{"name": "pos", "shape": [T, N, 3]}, ...], ...extra fields}.
// Python writes it in aevion/viewer/protocol.py; the two implementations are tested against each other.

import { b64ToF32 } from '../core/util.js';

export const MAGIC_VERSION = 1;

export function decodeBundle(buffer) {
  const buf = buffer instanceof ArrayBuffer ? buffer : buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  if (buf.byteLength < 4) throw new Error('bundle too short');
  const view = new DataView(buf);
  const hlen = view.getUint32(0, true);
  if (4 + hlen > buf.byteLength) throw new Error('bundle header exceeds the buffer');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  let off = 4 + hlen;
  off = (off + 3) & ~3;
  const arrays = {};
  for (const a of header.arrays || []) {
    const n = a.shape.reduce((p, q) => p * q, 1);
    if (!Number.isFinite(n) || n < 0 || off + n * 4 > buf.byteLength) throw new Error(`array '${a.name}' does not fit in the bundle`);
    arrays[a.name] = { data: new Float32Array(buf.slice(off, off + n * 4)), shape: a.shape };
    off += n * 4;
  }
  return { header, arrays };
}

export function encodeBundle(header, arrays) {
  const names = Object.keys(arrays);
  const h = { ...header, arrays: names.map((name) => ({ name, shape: arrays[name].shape })) };
  const hb = new TextEncoder().encode(JSON.stringify(h));
  const start = (4 + hb.length + 3) & ~3;
  let total = start;
  for (const n of names) total += arrays[n].data.length * 4;
  const out = new ArrayBuffer(total);
  new DataView(out).setUint32(0, hb.length, true);
  new Uint8Array(out, 4, hb.length).set(hb);
  let off = start;
  for (const n of names) {
    new Float32Array(out, off, arrays[n].data.length).set(arrays[n].data);
    off += arrays[n].data.length * 4;
  }
  return out;
}

/** Decode the base64 `{shape, data}` entries of a static payload into typed arrays. */
export function decodeChannels(entries) {
  const out = {};
  for (const [name, e] of Object.entries(entries || {})) out[name] = { data: b64ToF32(e.b64), shape: e.shape };
  return out;
}
