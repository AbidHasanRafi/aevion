// Data sources: the three ways the viewer receives a run.
//
//   StaticSource   everything embedded in the page (single-file replay, Jupyter)
//   ChunkedSource  HTTP API of `aevion view` (long runs: a coarse overview plus fine windows around the playhead)
//   LiveSource     WebSocket stream of a running simulation (`aevion live`)
//
// All three expose the same shape: { meta, frames: MultiResStore, world, layers, runs, events } plus `ready` / `onUpdate`.
// Network access (fetch, WebSocket) is injected so the classes are testable under Node.

import { Emitter } from '../core/util.js';
import { decodeBundle, decodeChannels } from './decode.js';
import { FrameStore, MultiResStore } from './frames.js';
import { b64ToF32 } from '../core/util.js';

export const FORMAT = 'aevion.view/2';

function checkFormat(p) {
  if (!p || typeof p !== 'object' || !p.meta) throw new Error('not an Aevion viewer payload');
  const f = p.format || 'aevion.view/1';
  if (f !== FORMAT && f !== 'aevion.view/1') throw new Error(`unsupported viewer payload format '${f}'`);
}

/** Build a FrameStore from a v2 payload object (v1 payloads without shapes are inferred from meta.T/n/M). */
export function storeFromPayload(p) {
  const { meta } = p;
  const N = meta.n, T = meta.T, M = meta.M;
  const shapes = meta.shapes || {
    pos: [T, N, 3], quat: [T, N, 4], vel: [T, N, 3], euler: [T, N, 3], rotor: [T, N, M], thrust: [T, N], power: [T, N], soc: [T, N], wind: [T, N, 3],
    ref: [T, N, 3], err: [T, N],
  };
  const t = b64ToF32(p.t);
  if (t.length !== T) throw new Error(`time axis has ${t.length} entries, expected ${T}`);
  const arrays = {};
  for (const name of Object.keys(shapes)) {
    if (name === 't' || typeof p[name] !== 'string') continue;
    const data = b64ToF32(p[name]);
    const expect = shapes[name].reduce((a, b) => a * b, 1);
    if (data.length !== expect) throw new Error(`channel '${name}' has ${data.length} values, expected ${expect}`);
    arrays[name] = { data, shape: shapes[name] };
  }
  Object.assign(arrays, decodeChannels(p.extras));
  return FrameStore.fromArrays(t, arrays, N);
}

export class StaticSource extends Emitter {
  constructor(payload) {
    super();
    checkFormat(payload);
    this.kind = 'static';
    this.meta = payload.meta;
    this.world = payload.world || null;
    this.layers = { lidar: payload.lidar || null, wireless: payload.wireless || null, camera: payload.camera || null, planned: payload.planned || null, rem: payload.rem || null, beams: payload.beams || null, geofences: payload.geofences || null, actors: payload.actors || null };
    this.runs = payload.runs || [];
    this.events = payload.meta.events || [];
    this.frames = new MultiResStore(storeFromPayload(payload));
    this.ready = Promise.resolve(this);
    this.live = false;
  }
  setFocusTime() {}
}

/** Reads `aevion.view` data from an HTTP API: GET /api/meta and GET /api/frames. */
export class ChunkedSource extends Emitter {
  constructor(baseUrl = '', fetchImpl = (...a) => fetch(...a), { overviewFrames = 4000, detailFrames = 1500, detailSpan = null } = {}) {
    super();
    this.kind = 'chunked'; this.base = baseUrl.replace(/\/$/, ''); this.fetch = fetchImpl;
    this.overviewFrames = overviewFrames; this.detailFrames = detailFrames; this.detailSpan = detailSpan;
    this.live = false; this._pending = null; this.ready = this.open();
  }

  async _json(path) {
    const r = await this.fetch(this.base + path);
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  }

  async _frames(query) {
    const r = await this.fetch(this.base + '/api/frames?' + new URLSearchParams(query));
    if (!r.ok) throw new Error(`frames: HTTP ${r.status}`);
    return decodeBundle(await r.arrayBuffer());
  }

  _store(bundle, N) {
    const t0 = bundle.header.t0 || 0, dt = bundle.arrays.t.data, t = new Float64Array(dt.length);
    for (let i = 0; i < dt.length; i++) t[i] = t0 + dt[i];
    const arrays = { ...bundle.arrays }; delete arrays.t;
    return FrameStore.fromArrays(t, arrays, N);
  }

  async open() {
    const m = await this._json('/api/meta');
    checkFormat(m);
    this.meta = m.meta; this.world = m.world || null; this.runs = m.runs || []; this.events = m.meta.events || [];
    this.layers = { lidar: m.lidar || null, wireless: m.wireless || null, camera: m.camera || null, planned: m.planned || null, rem: m.rem || null, beams: m.beams || null, geofences: m.geofences || null, actors: m.actors || null };
    this.names = m.channels; this.t0 = m.t0; this.t1 = m.t1;
    const ov = await this._frames({ names: this.names.join(','), max: this.overviewFrames });
    this.frames = new MultiResStore(this._store(ov, m.meta.n));
    return this;
  }

  /** Make sure a fine-resolution window covers `time`; resolves when the window has been swapped in. */
  async ensureDetail(time) {
    const f = this.frames, N = this.meta.n, dur = this.t1 - this.t0;
    const span = this.detailSpan || Math.max(dur / 12, 1);
    if (f.detail && time >= f.detail.t0 + span * 0.1 && time <= f.detail.t1 - span * 0.1) return false;
    if (this._pending) return false;
    const start = Math.max(this.t0, time - span * 0.25), stop = Math.min(this.t1, start + span);
    this._pending = this._frames({ names: this.names.join(','), start, stop, max: this.detailFrames })
      .then((b) => { f.setDetail(this._store(b, N)); this.emit('detail', start, stop); return true; })
      .finally(() => { this._pending = null; });
    return this._pending;
  }
}

/** Receives frames over a WebSocket: a JSON `meta` message, then binary `frames` bundles, then `end`. */
export class LiveSource extends Emitter {
  constructor(url, WS = globalThis.WebSocket, { reconnect = false } = {}) {
    super();
    this.kind = 'live'; this.live = true; this.url = url; this.WS = WS; this.reconnect = reconnect;
    this.frames = null; this.meta = null; this.world = null; this.layers = {}; this.runs = []; this.events = [];
    this.finished = false; this.closed = false;
    this.ready = new Promise((res, rej) => { this._res = res; this._rej = rej; });
    this._connect();
  }

  _connect() {
    const ws = (this.ws = new this.WS(this.url));
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (ev) => this._message(ev.data);
    ws.onerror = () => { this.emit('error', new Error('websocket error')); if (!this.meta) this._rej(new Error('websocket error')); };
    ws.onclose = () => { this.closed = true; this.emit('close'); if (this.reconnect && !this.finished) setTimeout(() => this._connect(), 1000); };
  }

  _message(data) {
    if (typeof data === 'string') {
      const m = JSON.parse(data);
      if (m.type === 'meta') {
        checkFormat(m);
        this.meta = m.meta; this.world = m.world || null; this.events = m.meta.events || []; this.runs = m.runs || [];
        this.layers = { lidar: null, wireless: m.wireless || null, camera: m.camera || null, planned: m.planned || null, rem: m.rem || null, beams: m.beams || null, geofences: m.geofences || null, actors: m.actors || null };
        this.frames = new MultiResStore(new FrameStore(m.meta.n, m.widths || {}, 512));
        this._res(this);
        this.emit('meta', this);
      } else if (m.type === 'event') { this.events.push(m.event); this.emit('event', m.event); }
      else if (m.type === 'end') { this.finished = true; this.emit('end', m); }
      return;
    }
    if (!this.frames) return; // frames before meta: ignore
    const { header, arrays } = decodeBundle(data);
    if (header.type !== 'frames') return;
    const times = arrays.t.data, t0 = header.t0 || 0, n = times.length;
    const abs = new Float64Array(n);
    for (let i = 0; i < n; i++) abs[i] = t0 + times[i];
    const rest = { ...arrays }; delete rest.t;
    this.frames.overview.appendMany(abs, rest);
    this.emit('frames', n);
  }

  close() { this.closed = true; this.reconnect = false; try { this.ws.close(); } catch { /* already closed */ } }
}
