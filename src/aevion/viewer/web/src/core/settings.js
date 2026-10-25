// Persistent viewer settings. Storage is optional (private windows, blocked storage and iframes can throw): every access is guarded.

import { Emitter } from './util.js';

export const DEFAULTS = {
  theme: 'dark',          // dark | light | print
  quality: 'auto',        // auto | low | medium | high
  timeOfDay: 13,          // hour, 0-24
  trailColor: 'speed',    // speed | altitude | sinr | soc | none
  trailSeconds: 0,        // 0 = whole path
  vehicleScale: 0,        // 0 = automatic
  followFocus: false,
  coverageMode: 'slice',  // slice | volume
  units: 'si',
  showHud: true,
  showCharts: true,
  chartChannels: null,
};

export class Settings extends Emitter {
  constructor(storage = null, key = 'aevion.viewer.v2') {
    super();
    this.key = key; this.values = {}; this.storage = storage;
    try { const raw = storage && storage.getItem(key); if (raw) this.values = JSON.parse(raw) || {}; } catch { this.values = {}; }
  }

  get(name) { return this.values[name] !== undefined ? this.values[name] : DEFAULTS[name]; }

  set(name, value) {
    if (this.values[name] === value) return;
    this.values[name] = value;
    try { this.storage && this.storage.setItem(this.key, JSON.stringify(this.values)); } catch { /* storage unavailable: keep in memory */ }
    this.emit('change', name, value);
    this.emit('change:' + name, value);
  }

  reset() { this.values = {}; try { this.storage && this.storage.removeItem(this.key); } catch { /* ignore */ } this.emit('reset'); }
}

/** localStorage if usable, else null. */
export function safeStorage() {
  try { const s = globalThis.localStorage; const k = '__aevion_probe__'; s.setItem(k, '1'); s.removeItem(k); return s; } catch { return null; }
}

/** Pick a rendering quality from a rough capability probe: `auto` -> low | medium | high. */
export function autoQuality({ cores = 4, deviceMemory = 4, renderer = '', nVehicles = 1 } = {}) {
  const r = renderer.toLowerCase();
  const integrated = /intel|uhd|iris|mali|adreno|apple gpu|llvmpipe|swiftshader|software/.test(r);
  if (/swiftshader|llvmpipe|software/.test(r)) return 'low';
  if (integrated || deviceMemory <= 4 || cores <= 4) return nVehicles > 30 ? 'low' : 'medium';
  return 'high';
}
