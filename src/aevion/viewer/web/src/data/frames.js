// Time-indexed frame storage with interpolation, growth (live streaming) and two resolution levels.
//
// A FrameStore holds a sorted time axis `t` (length T) and named channels laid out as (T, N, k): frame-major,
// then vehicle, then component. Channels can be static views onto decoded payload arrays or growable buffers
// that live frames are appended to. Everything here is pure (no DOM, no three.js) and unit-tested under Node.

import { clamp, lowerBound } from '../core/util.js';

/** Spherical linear interpolation of unit quaternions stored as [w, x, y, z]; writes into `out`. */
export function slerpWxyz(a, ao, b, bo, t, out) {
  let aw = a[ao], ax = a[ao + 1], ay = a[ao + 2], az = a[ao + 3];
  let bw = b[bo], bx = b[bo + 1], by = b[bo + 2], bz = b[bo + 3];
  let dot = aw * bw + ax * bx + ay * by + az * bz;
  if (dot < 0) { dot = -dot; bw = -bw; bx = -bx; by = -by; bz = -bz; }
  let s0, s1;
  if (dot > 0.9995) { s0 = 1 - t; s1 = t; } else {
    const th = Math.acos(clamp(dot, -1, 1)), sn = Math.sin(th);
    s0 = Math.sin((1 - t) * th) / sn; s1 = Math.sin(t * th) / sn;
  }
  const w = s0 * aw + s1 * bw, x = s0 * ax + s1 * bx, y = s0 * ay + s1 * by, z = s0 * az + s1 * bz;
  const n = Math.hypot(w, x, y, z) || 1;
  out[0] = w / n; out[1] = x / n; out[2] = y / n; out[3] = z / n;
  return out;
}

export class FrameStore {
  /**
   * @param {number} N number of vehicles
   * @param {{[name: string]: number}} widths components per vehicle for each channel (pos: 3, quat: 4, soc: 1, ...)
   * @param {number} capacity initial frame capacity for growable stores
   */
  constructor(N, widths = {}, capacity = 0) {
    this.N = N;
    this.widths = { ...widths };
    this.cap = capacity;
    this.T = 0;
    this.t = new Float64Array(capacity);
    this.ch = new Map();
    this.version = 0;
    this._range = new Map();
    for (const [name, k] of Object.entries(widths)) this.ch.set(name, new Float32Array(capacity * N * k));
  }

  /** Wrap decoded static arrays: `arrays[name] = {data, shape: [T, N, k]}` (a missing trailing dimension means k = 1). */
  static fromArrays(t, arrays, N) {
    const T = t.length;
    const s = new FrameStore(N, {}, 0);
    s.t = Float64Array.from(t);
    s.T = s.cap = T;
    for (const [name, a] of Object.entries(arrays)) {
      const k = a.shape.length >= 3 ? a.shape[2] : 1;
      if (a.shape[0] !== T) throw new Error(`channel '${name}' has ${a.shape[0]} frames, expected ${T}`);
      s.widths[name] = k;
      s.ch.set(name, a.data);
    }
    return s;
  }

  has(name) { return this.ch.has(name); }
  names() { return [...this.ch.keys()]; }
  get duration() { return this.T ? this.t[this.T - 1] - this.t[0] : 0; }
  get t0() { return this.T ? this.t[0] : 0; }
  get t1() { return this.T ? this.t[this.T - 1] : 0; }

  _grow(minCap) {
    if (minCap <= this.cap) return;
    const cap = Math.max(minCap, this.cap * 2, 256);
    const t = new Float64Array(cap); t.set(this.t.subarray(0, this.T)); this.t = t;
    for (const [name, k] of Object.entries(this.widths)) {
      const old = this.ch.get(name), n = new Float32Array(cap * this.N * k);
      n.set(old.subarray(0, this.T * this.N * k)); this.ch.set(name, n);
    }
    this.cap = cap;
  }

  /** Append one frame (live streaming). `frame[name]` is a typed array of N*k numbers; unknown channels are created on demand. */
  append(time, frame) {
    if (this.T && time <= this.t[this.T - 1]) return false; // time must increase
    for (const [name, v] of Object.entries(frame)) {
      if (!this.widths[name]) { this.widths[name] = Math.max(1, Math.round(v.length / this.N)); this.ch.set(name, new Float32Array(this.cap * this.N * this.widths[name])); }
    }
    this._grow(this.T + 1);
    this.t[this.T] = time;
    for (const [name, k] of Object.entries(this.widths)) {
      const v = frame[name];
      const dst = this.ch.get(name), o = this.T * this.N * k;
      if (v) dst.set(v.length === this.N * k ? v : v.subarray(0, this.N * k), o);
      else if (this.T > 0) dst.copyWithin(o, o - this.N * k, o); // hold the last value for channels missing from this frame
    }
    this.T += 1; this.version += 1; this._range.clear();
    return true;
  }

  /** Append many frames at once (a decoded chunk); `arrays[name]` has shape (n, N, k). */
  appendMany(times, arrays) {
    const n = times.length;
    for (let i = 0; i < n; i++) {
      const frame = {};
      for (const [name, a] of Object.entries(arrays)) {
        const k = a.shape.length >= 3 ? a.shape[2] : 1, w = this.N * k;
        frame[name] = a.data.subarray(i * w, (i + 1) * w);
      }
      this.append(times[i], frame);
    }
  }

  /** Series (length T) of one component of one vehicle, as a fresh Float32Array. */
  series(name, v = 0, j = 0) {
    const k = this.widths[name], d = this.ch.get(name), out = new Float32Array(this.T);
    for (let i = 0; i < this.T; i++) out[i] = d[(i * this.N + v) * k + j];
    return out;
  }

  /** [min, max] of a channel's component over all vehicles and frames (cached until the store grows). */
  range(name, j = 0) {
    const key = name + ':' + j;
    let r = this._range.get(key);
    if (r) return r;
    const k = this.widths[name], d = this.ch.get(name);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < this.T; i++) for (let v = 0; v < this.N; v++) { const x = d[(i * this.N + v) * k + j]; if (x < lo) lo = x; if (x > hi) hi = x; }
    r = Number.isFinite(lo) ? [lo, hi] : [0, 1];
    this._range.set(key, r);
    return r;
  }

  /** Index of the frame at or before `time` and the interpolation weight towards the next one. */
  locate(time, out = { i0: 0, i1: 0, a: 0 }) {
    const T = this.T;
    if (T === 0) { out.i0 = out.i1 = 0; out.a = 0; return out; }
    if (time <= this.t[0]) { out.i0 = out.i1 = 0; out.a = 0; return out; }
    if (time >= this.t[T - 1]) { out.i0 = out.i1 = T - 1; out.a = 0; return out; }
    const hi = lowerBound(this.t, time, T); // first t >= time
    out.i1 = hi; out.i0 = hi - 1;
    const d = this.t[out.i1] - this.t[out.i0];
    out.a = d > 0 ? (time - this.t[out.i0]) / d : 0;
    return out;
  }

  /** Linear interpolation of one component of one vehicle. */
  get(name, loc, v = 0, j = 0) {
    const k = this.widths[name], d = this.ch.get(name);
    if (!d) return NaN;
    const a = d[(loc.i0 * this.N + v) * k + j], b = d[(loc.i1 * this.N + v) * k + j];
    return a + (b - a) * loc.a;
  }

  /** Interpolated vector of `n` components into `out`. */
  vec(name, loc, v, out, n = this.widths[name]) {
    for (let j = 0; j < n; j++) out[j] = this.get(name, loc, v, j);
    return out;
  }

  /** Interpolated attitude quaternion [w, x, y, z] of vehicle v. */
  quat(loc, v, out = new Float64Array(4)) {
    const q = this.ch.get('quat');
    if (!q) { out[0] = 1; out[1] = out[2] = out[3] = 0; return out; }
    return slerpWxyz(q, (loc.i0 * this.N + v) * 4, q, (loc.i1 * this.N + v) * 4, loc.a, out);
  }

  /** Nearest frame index to `time`. */
  nearest(time) {
    const l = this.locate(time);
    return l.a >= 0.5 ? l.i1 : l.i0;
  }
}

/**
 * Two resolution levels: a coarse `overview` of the whole run and an optional fine `detail` window around the playhead.
 * `cursor.seek(t)` picks the detail store whenever it covers `t`.
 */
export class MultiResStore {
  constructor(overview, detail = null) { this.overview = overview; this.detail = detail; }
  get N() { return this.overview.N; }
  get duration() { return this.overview.duration; }
  get t0() { return this.overview.t0; }
  get t1() { return this.overview.t1; }
  setDetail(store) { this.detail = store; }
  pick(time) {
    const d = this.detail;
    return d && d.T > 1 && time >= d.t0 && time <= d.t1 ? d : this.overview;
  }
  has(name) { return this.overview.has(name); }
  cursor() { return new Cursor(this); }
}

/** A reusable playback position: seek once per frame, then read any channel without allocating. */
export class Cursor {
  constructor(source) {
    this.source = source instanceof FrameStore ? new MultiResStore(source) : source;
    this.loc = { i0: 0, i1: 0, a: 0 };
    this.store = this.source.overview;
    this.time = 0;
  }
  get N() { return this.source.N; }
  seek(time) {
    this.time = time;
    this.store = this.source.pick(time);
    this.store.locate(time, this.loc);
    return this;
  }
  get(name, v = 0, j = 0) { return this.store.has(name) ? this.store.get(name, this.loc, v, j) : NaN; }
  vec(name, v, out, n) { return this.store.has(name) ? this.store.vec(name, this.loc, v, out, n) : out; }
  quat(v, out) { return this.store.quat(this.loc, v, out); }
  has(name) { return this.source.has(name); }
  /** Frame index in the overview store (used for trail draw ranges and chart cursors). */
  get overviewIndex() { const l = this.source.overview.locate(this.time); return l.i0; }
}
