// Perceptually ordered colour maps (pure). Control points are sampled from well-known colour-blind-safe maps.

const MAPS = {
  // viridis (9 stops)
  viridis: [[0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530], [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551], [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]],
  // the original Aevion speed ramp: blue -> cyan -> green -> yellow -> orange
  speed: [[0.10, 0.14, 0.55], [0.0, 0.55, 0.85], [0.10, 0.78, 0.55], [0.95, 0.85, 0.20], [1.0, 0.45, 0.10]],
  // red (bad) -> amber -> green (good), for link quality
  quality: [[0.86, 0.15, 0.15], [0.96, 0.55, 0.12], [0.95, 0.85, 0.25], [0.45, 0.78, 0.35], [0.12, 0.65, 0.40]],
  // magma-like, for heat / interference
  heat: [[0.0, 0.0, 0.02], [0.23, 0.06, 0.43], [0.55, 0.16, 0.50], [0.87, 0.29, 0.40], [0.99, 0.62, 0.37], [0.99, 0.99, 0.75]],
  // blue-white-red, for signed differences
  diverging: [[0.17, 0.35, 0.75], [0.55, 0.70, 0.90], [0.96, 0.96, 0.96], [0.96, 0.65, 0.55], [0.75, 0.15, 0.15]],
};

export const COLORMAP_NAMES = Object.keys(MAPS);

/** RGB triple in [0, 1] for u in [0, 1]; out-of-range and NaN values clamp to the ends. */
export function sampleColormap(name, u, out = [0, 0, 0]) {
  const stops = MAPS[name] || MAPS.viridis;
  if (!(u === u)) u = 0;
  u = Math.min(Math.max(u, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(u), stops.length - 2), t = u - i, a = stops[i], b = stops[i + 1];
  out[0] = a[0] + (b[0] - a[0]) * t; out[1] = a[1] + (b[1] - a[1]) * t; out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}

/** Colour for a link SINR in dB (red below the outage threshold, green at 25 dB and above). */
export function sinrColor(db, outage = -6, out = [0, 0, 0]) {
  return sampleColormap('quality', (db - outage) / 31, out);
}

/** Precomputed 256-entry lookup table, as a flat Uint8Array of RGB bytes. */
export function lut(name, n = 256) {
  const out = new Uint8Array(n * 3), c = [0, 0, 0];
  for (let i = 0; i < n; i++) { sampleColormap(name, i / (n - 1), c); out[3 * i] = Math.round(c[0] * 255); out[3 * i + 1] = Math.round(c[1] * 255); out[3 * i + 2] = Math.round(c[2] * 255); }
  return out;
}
