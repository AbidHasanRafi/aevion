import test from 'node:test';
import assert from 'node:assert/strict';

import { b64ToF32, bytesToB64, fmtTime, niceTicks, lowerBound, parseHash, buildHash, esc, Emitter } from '../src/core/util.js';
import { decodeBundle, encodeBundle, decodeChannels } from '../src/data/decode.js';
import { FrameStore, MultiResStore, Cursor, slerpWxyz } from '../src/data/frames.js';
import { sampleColormap, sinrColor, lut, COLORMAP_NAMES } from '../src/data/colormaps.js';
import { channelInfo, chartCandidates } from '../src/data/channels.js';
import { valueAt, seriesOf, differenceSeries, divergenceTime } from '../src/data/derived.js';
import { StaticSource, ChunkedSource, LiveSource, storeFromPayload } from '../src/data/source.js';

const f32b64 = (arr) => bytesToB64(new Uint8Array(new Float32Array(arr).buffer));
const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// ------------------------------------------------------------------------------------------ util
test('base64 round trip of float32 arrays', () => {
  const a = [1.5, -2.25, 3.125, 1e-3, 123456];
  assert.deepEqual([...b64ToF32(f32b64(a))], [...new Float32Array(a)]);
  assert.equal(b64ToF32('').length, 0);
});

test('formatting helpers', () => {
  assert.equal(fmtTime(75.3), '1:15.3');
  assert.equal(fmtTime(3725), '1:02:05');
  assert.equal(fmtTime(0.5), '0.50');
  assert.equal(fmtTime(NaN), '–');
  assert.deepEqual(niceTicks(0, 10, 5), [0, 2, 4, 6, 8, 10]);
  assert.deepEqual(niceTicks(0.0, 1.0, 4), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepEqual(niceTicks(-1, 1, 4), [-1, -0.5, 0, 0.5, 1]);
  assert.deepEqual(niceTicks(3, 3), [3]);
  assert.equal(esc('<b>"a" & \'b\'</b>'), '&lt;b&gt;&quot;a&quot; &amp; &#39;b&#39;&lt;/b&gt;');
});

test('lowerBound is a binary search', () => {
  const a = [0, 1, 2, 2, 5];
  assert.equal(lowerBound(a, -1), 0); assert.equal(lowerBound(a, 2), 2); assert.equal(lowerBound(a, 3), 4); assert.equal(lowerBound(a, 9), 5);
  assert.equal(lowerBound(a, 2, 2), 2);
});

test('hash state round trip', () => {
  const h = buildHash({ t: 5.5, cam: 'chase', play: 0, empty: '' });
  assert.deepEqual(parseHash(h), { t: '5.5', cam: 'chase', play: '0' });
  assert.deepEqual(parseHash(''), {});
});

test('emitter', () => {
  const e = new Emitter(); const got = [];
  const off = e.on('x', (a) => got.push(a)); e.emit('x', 1); off(); e.emit('x', 2);
  assert.deepEqual(got, [1]);
});

// --------------------------------------------------------------------------------------- bundles
test('bundle encode/decode round trip with alignment', () => {
  const arrays = { t: { data: Float32Array.from([0, 0.1, 0.2]), shape: [3] }, pos: { data: Float32Array.from({ length: 18 }, (_, i) => i * 0.5), shape: [3, 2, 3] } };
  const buf = encodeBundle({ type: 'frames', t0: 12.5 }, arrays);
  assert.equal(buf.byteLength % 4, 0);
  const { header, arrays: out } = decodeBundle(buf);
  assert.equal(header.type, 'frames'); assert.equal(header.t0, 12.5);
  assert.deepEqual(out.pos.shape, [3, 2, 3]); assert.deepEqual([...out.pos.data], [...arrays.pos.data]); assert.deepEqual([...out.t.data], [...arrays.t.data]);
});

test('decodeBundle rejects truncated or inconsistent input', () => {
  assert.throws(() => decodeBundle(new ArrayBuffer(2)), /too short/);
  const buf = encodeBundle({ type: 'x' }, { a: { data: Float32Array.from([1, 2, 3, 4]), shape: [4] } });
  assert.throws(() => decodeBundle(buf.slice(0, buf.byteLength - 4)), /does not fit/);
  const bad = new Uint8Array(buf.slice(0)); new DataView(bad.buffer).setUint32(0, 1e9, true);
  assert.throws(() => decodeBundle(bad.buffer), /exceeds/);
});

test('decodeChannels', () => {
  const c = decodeChannels({ a: { b64: f32b64([1, 2, 3, 4]), shape: [2, 2] } });
  assert.deepEqual(c.a.shape, [2, 2]); assert.deepEqual([...c.a.data], [1, 2, 3, 4]);
  assert.deepEqual(decodeChannels(undefined), {});
});

// ----------------------------------------------------------------------------------------- frames
function makeStore(T = 5, N = 2) {
  const t = Array.from({ length: T }, (_, i) => i * 0.5);
  const pos = new Float32Array(T * N * 3), quat = new Float32Array(T * N * 4), soc = new Float32Array(T * N);
  for (let i = 0; i < T; i++) for (let v = 0; v < N; v++) {
    pos.set([i, 10 * v, 2 * i], (i * N + v) * 3);
    const ang = i * 0.4; quat.set([Math.cos(ang / 2), 0, 0, Math.sin(ang / 2)], (i * N + v) * 4);
    soc[i * N + v] = 1 - 0.1 * i;
  }
  return FrameStore.fromArrays(t, { pos: { data: pos, shape: [T, N, 3] }, quat: { data: quat, shape: [T, N, 4] }, soc: { data: soc, shape: [T, N] } }, N);
}

test('locate and interpolate', () => {
  const s = makeStore();
  const loc = s.locate(0.75);
  assert.equal(loc.i0, 1); assert.equal(loc.i1, 2); near(loc.a, 0.5);
  near(s.get('pos', loc, 0, 0), 1.5); near(s.get('pos', loc, 1, 1), 10); near(s.get('pos', loc, 0, 2), 3);
  near(s.get('soc', loc, 0), 0.85);
  const before = s.locate(-3), after = s.locate(99);
  assert.deepEqual([before.i0, before.i1, before.a], [0, 0, 0]); assert.deepEqual([after.i0, after.i1, after.a], [4, 4, 0]);
  assert.equal(s.duration, 2); assert.equal(s.nearest(0.8), 2); assert.equal(s.nearest(0.7), 1);
});

test('non-uniform time axis (level of detail) is handled', () => {
  const s = FrameStore.fromArrays([0, 1, 10], { pos: { data: Float32Array.from([0, 0, 0, 1, 1, 1, 10, 10, 10]), shape: [3, 1, 3] } }, 1);
  near(s.get('pos', s.locate(5.5), 0, 0), 1 + (10 - 1) * (4.5 / 9));
});

test('slerp keeps unit norm and the shortest path', () => {
  const a = [1, 0, 0, 0, Math.cos(1.5), 0, 0, Math.sin(1.5)], out = [0, 0, 0, 0];
  slerpWxyz(a, 0, a, 4, 0.5, out);
  near(Math.hypot(...out), 1); near(2 * Math.atan2(out[3], out[0]), 1.5); // half of the 3.0 rad rotation
  // q and -q are the same rotation: interpolation must not go the long way round
  const neg = [1, 0, 0, 0, -Math.cos(0.05), 0, 0, -Math.sin(0.05)];
  slerpWxyz(neg, 0, neg, 4, 0.5, out);
  near(2 * Math.atan2(Math.abs(out[3]), Math.abs(out[0])), 0.05, 1e-9);
  slerpWxyz([1, 0, 0, 0], 0, [1, 0, 0, 0], 0, 0.3, out); assert.deepEqual([...out], [1, 0, 0, 0]);
});

test('store.quat interpolates attitude', () => {
  const s = makeStore();
  const q = s.quat(s.locate(0.25), 0);
  near(Math.hypot(...q), 1); near(2 * Math.atan2(q[3], q[0]), 0.2, 1e-5);
});

test('append grows the store and holds missing channels', () => {
  const s = new FrameStore(2, { pos: 3 }, 2);
  for (let i = 0; i < 700; i++) assert.ok(s.append(i * 0.1, { pos: Float32Array.from([i, i, i, -i, -i, -i]) }));
  assert.equal(s.T, 700); assert.ok(s.cap >= 700);
  near(s.get('pos', s.locate(35.05), 1, 0), -350.5, 1e-3);
  assert.equal(s.append(1.0, { pos: new Float32Array(6) }), false); // time must increase
  s.append(70.0, { pos: Float32Array.from([1, 2, 3, 4, 5, 6]), soc: Float32Array.from([0.5, 0.4]) });
  assert.deepEqual([s.widths.soc, s.get('soc', s.locate(70.0), 1, 0)], [1, Math.fround(0.4)]);
  s.append(71.0, { pos: Float32Array.from([1, 2, 3, 4, 5, 6]) }); // soc missing: held
  near(s.get('soc', s.locate(71.0), 1, 0), 0.4, 1e-6);
});

test('range cache is invalidated by growth', () => {
  const s = new FrameStore(1, { pos: 3 });
  s.append(0, { pos: Float32Array.from([0, 0, 1]) }); assert.deepEqual(s.range('pos', 2), [1, 1]);
  s.append(1, { pos: Float32Array.from([0, 0, 5]) }); assert.deepEqual(s.range('pos', 2), [1, 5]);
});

test('multi-resolution cursor prefers the detail window', () => {
  const overview = FrameStore.fromArrays([0, 10], { pos: { data: Float32Array.from([0, 0, 0, 10, 0, 0]), shape: [2, 1, 3] } }, 1);
  const detail = FrameStore.fromArrays([4, 5, 6], { pos: { data: Float32Array.from([4, 1, 0, 5, 1, 0, 6, 1, 0]), shape: [3, 1, 3] } }, 1);
  const ms = new MultiResStore(overview), c = new Cursor(ms);
  near(c.seek(5).get('pos', 0, 1), 0);
  ms.setDetail(detail);
  near(c.seek(5).get('pos', 0, 1), 1); near(c.get('pos', 0, 0), 5);
  near(c.seek(8).get('pos', 0, 1), 0); // outside the detail window: coarse data again
  assert.equal(new Cursor(overview).N, 1);
});

// --------------------------------------------------------------------------------------- colours
test('colormaps clamp, interpolate and are monotone in luminance for viridis', () => {
  for (const name of COLORMAP_NAMES) {
    for (const u of [-1, 0, 0.5, 1, 2, NaN]) { const c = sampleColormap(name, u); assert.ok(c.every((x) => x >= 0 && x <= 1), `${name} ${u}`); }
  }
  const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  let prev = -1; for (let i = 0; i <= 20; i++) { const l = lum(sampleColormap('viridis', i / 20)); assert.ok(l >= prev - 1e-9); prev = l; }
  assert.ok(sinrColor(-20)[0] > sinrColor(30)[0] && sinrColor(30)[1] > sinrColor(-20)[1]); // red when bad, green when good
  const t = lut('viridis', 16); assert.equal(t.length, 48);
});

test('channel info', () => {
  assert.deepEqual([channelInfo('err').unit, channelInfo('err').scale], ['cm', 100]);
  assert.equal(channelInfo('x:link_sinr_db').label, 'SINR');
  assert.equal(channelInfo('x:link_custom_thing').label, 'Link custom thing');
  assert.equal(channelInfo('x:my_signal').label, 'My signal');
  const s = FrameStore.fromArrays([0, 1], { 'x:b': { data: Float32Array.from([1, 2]), shape: [2, 1] }, 'x:a': { data: Float32Array.from([1, 2]), shape: [2, 1] } }, 1);
  assert.deepEqual(chartCandidates(s), ['speed', 'altitude', 'x:a', 'x:b']);
});

// -------------------------------------------------------------------------------------- derived
test('derived channels and series', () => {
  const T = 3, vel = Float32Array.from([3, 4, 0, 0, 0, 2, 1, 2, 2]), pos = Float32Array.from([0, 0, 5, 0, 0, 6, 0, 0, 7]);
  const s = FrameStore.fromArrays([0, 1, 2], { vel: { data: vel, shape: [T, 1, 3] }, pos: { data: pos, shape: [T, 1, 3] } }, 1);
  assert.deepEqual([...seriesOf(s, 'speed')], [5, 2, 3]);
  assert.deepEqual([...seriesOf(s, 'altitude')], [5, 6, 7]);
  assert.ok(Number.isNaN(seriesOf(s, 'wind_speed')[0]) && Number.isNaN(seriesOf(s, 'nothing')[0]));
  const c = new Cursor(s).seek(0.5);
  near(valueAt(c, 'speed'), Math.hypot(1.5, 2, 1)); near(valueAt(c, 'altitude'), 5.5); assert.ok(Number.isNaN(valueAt(c, 'nothing')));
});

test('difference between two runs and divergence time', () => {
  const mk = (off) => FrameStore.fromArrays([0, 1, 2, 3], { pos: { data: Float32Array.from([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0].map((x, i) => (i % 3 === 1 ? off * Math.floor(i / 3) : x))), shape: [4, 1, 3] } }, 1);
  const a = mk(0), b = mk(0.5), d = differenceSeries(a, b);
  assert.deepEqual([...d.d].map((x) => +x.toFixed(3)), [0, 0.5, 1, 1.5]);
  near(d.max, 1.5); near(d.rms, Math.sqrt((0 + 0.25 + 1 + 2.25) / 4));
  assert.equal(divergenceTime(d, 0.9), 2); assert.equal(divergenceTime(d, 5), null);
  // different sampling: b interpolated at a's times
  const coarse = FrameStore.fromArrays([0, 3], { pos: { data: Float32Array.from([0, 0, 0, 3, 0, 0]), shape: [2, 1, 3] } }, 1);
  near(differenceSeries(a, coarse).max, 0, 1e-6);
});

// --------------------------------------------------------------------------------------- sources
function payload(T = 4, N = 1, M = 4) {
  const rows = (k) => Array.from({ length: T * N * k }, (_, i) => i * 0.25);
  const p = { format: 'aevion.view/2', meta: { n: N, T, M, title: 'x', events: [{ t: 1, type: 'fault.start' }], shapes: { pos: [T, N, 3], quat: [T, N, 4], soc: [T, N] } },
    t: f32b64(Array.from({ length: T }, (_, i) => i * 0.1)), pos: f32b64(rows(3)), quat: f32b64(Array.from({ length: T * N * 4 }, (_, i) => (i % 4 === 0 ? 1 : 0))), soc: f32b64(rows(1)),
    extras: { 'x:link_sinr_db': { b64: f32b64(rows(1)), shape: [T, N] } }, world: { boxes: [] } };
  return p;
}

test('static source decodes a payload and validates shapes', () => {
  const src = new StaticSource(payload());
  assert.equal(src.kind, 'static'); assert.equal(src.frames.overview.T, 4); assert.ok(src.frames.has('x:link_sinr_db'));
  assert.equal(src.events.length, 1); assert.deepEqual(src.world, { boxes: [] });
  const bad = payload(); bad.pos = f32b64([1, 2, 3]);
  assert.throws(() => new StaticSource(bad), /pos/);
  assert.throws(() => new StaticSource({ meta: { n: 1 }, format: 'aevion.view/9' }), /unsupported/);
  assert.throws(() => new StaticSource(null), /not an Aevion/);
  const short = payload(); short.meta.T = 5;
  assert.throws(() => storeFromPayload(short), /time axis/);
});

test('v1 payloads without shapes are still readable', () => {
  const p = payload(); delete p.meta.shapes; delete p.format; delete p.extras;
  p.vel = f32b64(Array.from({ length: 12 }, () => 0)); p.euler = f32b64(Array.from({ length: 12 }, () => 0));
  p.rotor = f32b64(Array.from({ length: 16 }, () => 1)); p.thrust = f32b64([1, 1, 1, 1]); p.power = f32b64([1, 1, 1, 1]); p.wind = f32b64(Array.from({ length: 12 }, () => 0));
  const s = new StaticSource(p).frames.overview;
  assert.equal(s.widths.rotor, 4); assert.equal(s.widths.pos, 3);
});

function fakeFetch(routes) {
  return async (url) => {
    const u = new URL(url, 'http://x');
    const r = routes[u.pathname];
    if (!r) return { ok: false, status: 404 };
    const body = typeof r === 'function' ? r(u.searchParams) : r;
    return { ok: true, status: 200, json: async () => body, arrayBuffer: async () => body };
  };
}

test('chunked source loads an overview and swaps in detail windows', async () => {
  const mkBundle = (t0, n, y, step = 1) => encodeBundle({ type: 'frames', t0 }, { t: { data: Float32Array.from({ length: n }, (_, i) => i * step), shape: [n] }, pos: { data: Float32Array.from({ length: n * 3 }, (_, i) => (i % 3 === 1 ? y : i)), shape: [n, 1, 3] } });
  const calls = [];
  const f = fakeFetch({
    '/api/meta': { format: 'aevion.view/2', meta: { n: 1, T: 100, M: 4, title: 'long' }, world: null, channels: ['pos'], t0: 0, t1: 99 },
    '/api/frames': (q) => { calls.push(Object.fromEntries(q)); return q.has('start') ? mkBundle(Number(q.get('start')), 5, 7, 5) : mkBundle(0, 3, 0); },
  });
  const src = new ChunkedSource('http://x', f, { detailSpan: 20 });
  await src.ready;
  assert.equal(src.frames.overview.T, 3); assert.equal(src.meta.title, 'long');
  assert.equal(await src.ensureDetail(50), true);
  assert.equal(src.frames.detail.T, 5); near(src.frames.detail.t0, 45);
  assert.equal(await src.ensureDetail(52), false); // already covered
  assert.deepEqual(calls[0], { names: 'pos', max: '4000' }); assert.equal(calls[1].start, '45');
  const bad = new ChunkedSource('http://x', fakeFetch({}));
  await assert.rejects(bad.ready, /HTTP 404/);
});

class FakeWS { constructor(url) { FakeWS.last = this; this.url = url; } close() { this.closed = true; } }

test('live source builds a growing store from websocket messages', async () => {
  const live = new LiveSource('ws://x', FakeWS);
  const ws = FakeWS.last, got = [];
  live.on('frames', (n) => got.push(n));
  ws.onmessage({ data: encodeBundle({ type: 'frames', t0: 0 }, { t: { data: Float32Array.of(0), shape: [1] }, pos: { data: Float32Array.of(1, 2, 3), shape: [1, 1, 3] } }) }); // before meta: ignored
  assert.equal(live.frames, null);
  ws.onmessage({ data: JSON.stringify({ type: 'meta', format: 'aevion.view/2', meta: { n: 1, T: 0, M: 4, title: 'live', events: [] }, widths: { pos: 3 } }) });
  await live.ready;
  for (let k = 0; k < 3; k++) ws.onmessage({ data: encodeBundle({ type: 'frames', t0: 100 + k }, { t: { data: Float32Array.of(0, 0.5), shape: [2] }, pos: { data: Float32Array.of(k, 0, 0, k + 0.5, 0, 0), shape: [2, 1, 3] } }) });
  assert.equal(live.frames.overview.T, 6); assert.deepEqual(got, [2, 2, 2]);
  near(live.frames.overview.get('pos', live.frames.overview.locate(101.25), 0, 0), 1.25);
  ws.onmessage({ data: JSON.stringify({ type: 'event', event: { t: 101, type: 'fault.start' } }) });
  assert.equal(live.events.length, 1);
  let ended = false; live.on('end', () => { ended = true; });
  ws.onmessage({ data: JSON.stringify({ type: 'end' }) });
  assert.ok(ended && live.finished);
  live.close(); assert.ok(ws.closed);
});
