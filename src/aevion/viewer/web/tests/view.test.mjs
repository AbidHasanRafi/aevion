import test from 'node:test';
import assert from 'node:assert/strict';

import { Settings, DEFAULTS, autoQuality } from '../src/core/settings.js';
import { fmtTick, niceTicks } from '../src/core/util.js';
import { decimateMinMax, paddedRange, runs } from '../src/data/decimate.js';
import { channelInfo, chartCandidates } from '../src/data/channels.js';
import { FrameStore, Cursor } from '../src/data/frames.js';
import { seriesOf, valueAt } from '../src/data/derived.js';
import { zoneExtent } from '../src/data/zones.js';

class MemoryStorage {
  constructor(initial = {}) { this.d = { ...initial }; }
  getItem(k) { return k in this.d ? this.d[k] : null; }
  setItem(k, v) { this.d[k] = String(v); }
  removeItem(k) { delete this.d[k]; }
}

test('settings fall back to defaults and persist changes', () => {
  const st = new MemoryStorage();
  const s = new Settings(st);
  assert.equal(s.get('theme'), DEFAULTS.theme);
  s.set('theme', 'print');
  assert.equal(new Settings(st).get('theme'), 'print');
  s.reset();
  assert.equal(new Settings(st).get('theme'), DEFAULTS.theme);
});

test('settings survive corrupt or unavailable storage', () => {
  assert.equal(new Settings(new MemoryStorage({ 'aevion.viewer.v2': '{not json' })).get('quality'), 'auto');
  const broken = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); }, removeItem() { throw new Error('denied'); } };
  const s = new Settings(broken);
  s.set('quality', 'low');
  assert.equal(s.get('quality'), 'low');          // kept in memory
  assert.doesNotThrow(() => s.reset());
});

test('settings emit change events once per real change', () => {
  const s = new Settings(null); const seen = [];
  s.on('change:timeOfDay', (v) => seen.push(v));
  s.set('timeOfDay', 18); s.set('timeOfDay', 18); s.set('timeOfDay', 6);
  assert.deepEqual(seen, [18, 6]);
});

test('automatic quality follows the hardware', () => {
  assert.equal(autoQuality({ renderer: 'SwiftShader' }), 'low');
  assert.equal(autoQuality({ renderer: 'Intel(R) UHD Graphics', nVehicles: 100 }), 'low');
  assert.equal(autoQuality({ renderer: 'Intel(R) UHD Graphics', nVehicles: 3 }), 'medium');
  assert.equal(autoQuality({ renderer: 'NVIDIA GeForce RTX', cores: 16, deviceMemory: 16 }), 'high');
});

test('tick labels carry just enough decimals', () => {
  assert.equal(fmtTick(59.99, 0.01), '59.99');
  assert.equal(fmtTick(60, 0.01), '60.00');
  assert.equal(fmtTick(5000, 5000), '5000');
  assert.equal(fmtTick(0, 0.5), '0');
  assert.equal(fmtTick(2.5e5, 1e5), '2.5e5');
  assert.equal(fmtTick(NaN, 1), '–');
  const ticks = niceTicks(59.98, 60.01, 3), step = ticks[1] - ticks[0];
  assert.equal(new Set(ticks.map((t) => fmtTick(t, step))).size, ticks.length, 'neighbouring ticks must be distinguishable');
});

test('min/max decimation keeps every peak', () => {
  const n = 10000, t = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { t[i] = i; y[i] = Math.sin(i / 50); }
  y[4321] = 9;                                    // a one-sample spike must survive 100x reduction
  const d = decimateMinMax(t, y, 0, n - 1, 100);
  assert.ok(d.n <= 100 && Math.max(...d.hi) === 9);
  const sub = decimateMinMax(t, y, 4300, 4340, 40);
  assert.equal(Math.max(...sub.hi), 9);
});

test('decimation skips NaN and handles empty windows', () => {
  const t = [0, 1, 2, 3], y = [1, NaN, 3, NaN];
  const d = decimateMinMax(t, y, 0, 3, 4);
  assert.ok(Array.from(d.hi).every(Number.isFinite));
  assert.equal(decimateMinMax([], [], 0, 1, 10).n, 0);
});

test('padded range widens constant series and ignores NaN', () => {
  assert.deepEqual(paddedRange([0, 1], [5, 5], 0, 1), [4 - 0.16, 6 + 0.16]);
  assert.deepEqual(paddedRange([0, 1], [NaN, NaN], 0, 1), [0, 1]);
});

test('runs finds contiguous flagged stretches', () => {
  const f = [0, 1, 1, 0, 0, 1, 0, 1, 1].map(Boolean);
  assert.deepEqual(runs((i) => f[i], f.length), [[1, 3], [5, 6], [7, 9]]);
  assert.deepEqual(runs(() => false, 5), []);
  assert.deepEqual(runs(() => true, 3), [[0, 3]]);
});

test('channel labels, scaling and the chart picker', () => {
  assert.deepEqual(channelInfo('x:link_sinr_db'), { scale: 1, label: 'SINR', unit: 'dB' });
  assert.equal(channelInfo('err').scale, 100);
  assert.equal(channelInfo('x:something_new').label, 'Something new');
  const store = FrameStore.fromArrays([0, 1], { pos: { data: new Float32Array(6), shape: [2, 1, 3] }, 'x:link_sinr_db': { data: new Float32Array(2), shape: [2, 1] } }, 1);
  const c = chartCandidates(store);
  assert.ok(c.includes('speed') && c.includes('altitude') && c.includes('x:link_sinr_db'));
});

test('derived channels: speed and altitude from stored ones', () => {
  const vel = new Float32Array([3, 4, 0, 0, 0, 2]), pos = new Float32Array([0, 0, 10, 0, 0, 12]);
  const s = FrameStore.fromArrays([0, 1], { vel: { data: vel, shape: [2, 1, 3] }, pos: { data: pos, shape: [2, 1, 3] } }, 1);
  assert.deepEqual(Array.from(seriesOf(s, 'speed', 0)), [5, 2]);
  assert.deepEqual(Array.from(seriesOf(s, 'altitude', 0)), [10, 12]);
  const cur = new Cursor(s).seek(0.5);
  assert.ok(Math.abs(valueAt(cur, 'altitude', 0) - 11) < 1e-6);
});

test('geofence zones get a visible vertical extent when their ends are open', () => {
  assert.deepEqual(zoneExtent({ z_min: 10, z_max: 50 }, 100), [10, 50]);
  assert.deepEqual(zoneExtent({ z_min: null, z_max: 80 }, 100), [0, 80]);
  const [lo, hi] = zoneExtent({}, 200);
  assert.equal(lo, 0); assert.ok(hi >= 120);
  const [l2, h2] = zoneExtent({ z_min: 500, z_max: null }, 10);
  assert.ok(h2 > l2);                                   // never an empty volume
});
