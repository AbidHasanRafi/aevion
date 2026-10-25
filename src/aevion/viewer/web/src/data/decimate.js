// Chart decimation (pure): reduce a long series to at most `width` min/max columns so no peak is lost when drawing.

import { lowerBound } from '../core/util.js';

/**
 * @param {ArrayLike<number>} t sorted times
 * @param {ArrayLike<number>} y values (NaN entries are skipped)
 * @returns {{x: Float32Array, lo: Float32Array, hi: Float32Array, n: number}} per-column pixel x, min and max
 */
export function decimateMinMax(t, y, t0, t1, width, n = t.length) {
  const cols = Math.max(1, Math.floor(width)), x = new Float32Array(cols), lo = new Float32Array(cols), hi = new Float32Array(cols);
  const i0 = Math.max(lowerBound(t, t0, n) - 1, 0), i1 = Math.min(lowerBound(t, t1, n) + 1, n);
  let m = 0, cur = -1;
  const span = t1 - t0 || 1;
  for (let i = i0; i < i1; i++) {
    const v = y[i]; if (!Number.isFinite(v)) continue;
    const c = Math.min(Math.max(Math.floor(((t[i] - t0) / span) * cols), 0), cols - 1);
    if (c !== cur) { cur = c; x[m] = c + 0.5; lo[m] = v; hi[m] = v; m++; } else { if (v < lo[m - 1]) lo[m - 1] = v; if (v > hi[m - 1]) hi[m - 1] = v; }
  }
  return { x: x.subarray(0, m), lo: lo.subarray(0, m), hi: hi.subarray(0, m), n: m };
}

/** Finite min/max of a series over a window, padded by 8 % (and widened when constant). */
export function paddedRange(t, y, t0, t1, n = t.length) {
  let lo = Infinity, hi = -Infinity;
  const i0 = Math.max(lowerBound(t, t0, n) - 1, 0), i1 = Math.min(lowerBound(t, t1, n) + 1, n);
  for (let i = i0; i < i1; i++) { const v = y[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

/** Contiguous index ranges [start, end) where `flag(i)` is true, in time order. */
export function runs(flag, n) {
  const out = []; let s = -1;
  for (let i = 0; i < n; i++) { const f = flag(i); if (f && s < 0) s = i; else if (!f && s >= 0) { out.push([s, i]); s = -1; } }
  if (s >= 0) out.push([s, n]);
  return out;
}
