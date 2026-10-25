// Channels computed from the stored ones, and the difference between two runs (pure).

export const DERIVED = {
  speed: { needs: ['vel'], value: (c, v) => Math.hypot(c.get('vel', v, 0), c.get('vel', v, 1), c.get('vel', v, 2)) },
  altitude: { needs: ['pos'], value: (c, v) => c.get('pos', v, 2) },
  wind_speed: { needs: ['wind'], value: (c, v) => Math.hypot(c.get('wind', v, 0), c.get('wind', v, 1), c.get('wind', v, 2)) },
  tilt: { needs: ['euler'], value: (c, v) => Math.hypot(c.get('euler', v, 0), c.get('euler', v, 1)) * 180 / Math.PI },
};

/** Value of a stored or derived channel at the cursor for vehicle v (NaN if unavailable). */
export function valueAt(cursor, name, v = 0) {
  const d = DERIVED[name];
  if (d) return d.needs.every((n) => cursor.has(n)) ? d.value(cursor, v) : NaN;
  return cursor.has(name) ? cursor.get(name, v, 0) : NaN;
}

/** Full series (one value per stored frame) of a stored or derived channel for vehicle v. */
export function seriesOf(store, name, v = 0) {
  const d = DERIVED[name];
  const T = store.T, out = new Float32Array(T);
  if (!d) { if (!store.has(name)) return out.fill(NaN); return store.series(name, v, 0); }
  if (!d.needs.every((n) => store.has(n))) return out.fill(NaN);
  const loc = { i0: 0, i1: 0, a: 0 };
  const cur = { get: (n, vv, j) => store.get(n, loc, vv, j) };
  for (let i = 0; i < T; i++) { loc.i0 = loc.i1 = i; out[i] = d.value(cur, v); }
  return out;
}

/**
 * Position difference |a(t) - b(t)| between two runs on the time axis of `a`; b is interpolated and held at its ends.
 * Returns { t, d, max, rms } with distances in the stores' length unit.
 */
export function differenceSeries(a, b, va = 0, vb = 0) {
  const T = a.T, t = new Float64Array(T), d = new Float32Array(T);
  const loc = { i0: 0, i1: 0, a: 0 };
  let mx = 0, ss = 0;
  for (let i = 0; i < T; i++) {
    const ti = a.t[i]; t[i] = ti;
    b.locate(ti, loc);
    const dx = a.ch.get('pos')[(i * a.N + va) * 3] - b.get('pos', loc, vb, 0);
    const dy = a.ch.get('pos')[(i * a.N + va) * 3 + 1] - b.get('pos', loc, vb, 1);
    const dz = a.ch.get('pos')[(i * a.N + va) * 3 + 2] - b.get('pos', loc, vb, 2);
    const e = Math.hypot(dx, dy, dz);
    d[i] = e; if (e > mx) mx = e; ss += e * e;
  }
  return { t, d, max: mx, rms: Math.sqrt(ss / Math.max(T, 1)) };
}

/** Time of the first sample whose difference exceeds `threshold` (the "divergence time"), or null. */
export function divergenceTime(diff, threshold) {
  for (let i = 0; i < diff.d.length; i++) if (diff.d[i] > threshold) return diff.t[i];
  return null;
}
