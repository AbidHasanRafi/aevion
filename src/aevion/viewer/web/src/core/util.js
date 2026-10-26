// Small, dependency-free helpers shared by the viewer (pure: no DOM, no three.js).

export const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
export const lerp = (a, b, t) => a + (b - a) * t;
export const DEG = 180 / Math.PI;
export const RAD = Math.PI / 180;

/** Decode a base64 string to a Uint8Array (works in browsers and Node >= 16). */
export function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Base64 string -> Float32Array (little-endian, as written by NumPy on every supported platform). */
export function b64ToF32(b64) {
  const u8 = b64ToBytes(b64);
  return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength >> 2);
}

export function bytesToB64(u8) {
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
  return btoa(s);
}

/** Format a number with a fixed number of decimals; non-finite values print as an en dash. */
export function fmt(x, digits = 2) {
  return Number.isFinite(x) ? x.toFixed(digits) : '–';
}

/** Human-readable duration: 75.3 -> "1:15.3", 3725 -> "1:02:05". */
export function fmtTime(t) {
  if (!Number.isFinite(t)) return '–';
  const sign = t < 0 ? '-' : '';
  t = Math.abs(t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  if (h > 0) return `${sign}${h}:${String(m).padStart(2, '0')}:${String(Math.floor(s)).padStart(2, '0')}`;
  if (m > 0) return `${sign}${m}:${s.toFixed(1).padStart(4, '0')}`;
  return `${sign}${s.toFixed(2)}`;
}

/** "Nice" tick positions for an axis from lo to hi (about `count` ticks). */
export function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(count, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+(Math.round(v / step) * step).toPrecision(12));
  return out;
}

/** Label for a tick value on an axis whose tick spacing is `step`: just enough decimals to tell neighbouring ticks apart. */
export function fmtTick(v, step) {
  if (!Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  if (Math.abs(v) >= 1e5 || (step > 0 && step < 1e-4)) return v.toExponential(1).replace('e+', 'e');
  const decimals = step > 0 ? Math.min(Math.max(Math.ceil(-Math.log10(step) - 1e-9), 0), 6) : 2;
  return v.toFixed(decimals);
}

/** Index of the first element >= x in a sorted array (binary search); returns arr.length if none. */
export function lowerBound(arr, x, n = arr.length) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** Deep-freeze-free shallow merge of defaults and overrides (only known keys of `defaults` are taken). */
export function withDefaults(defaults, given = {}) {
  const out = { ...defaults };
  for (const k of Object.keys(defaults)) if (given[k] !== undefined) out[k] = given[k];
  return out;
}

/** Parse "#t=5&cam=chase&play=0" style deep-link state into an object. */
export function parseHash(hash) {
  const out = {};
  const q = new URLSearchParams((hash || '').replace(/^#/, ''));
  for (const [k, v] of q) out[k] = v;
  return out;
}

export function buildHash(state) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(state)) if (v !== undefined && v !== null && v !== '') q.set(k, String(v));
  return '#' + q.toString();
}

/** Escape text for safe insertion in innerHTML (titles and event names come from data files). */
export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Font for canvas text: the interface face, at the 10 px minimum used for chart labels. */
export const CANVAS_FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';

/** Colours of the active theme (read from the CSS tokens once per theme, not once per chart per frame). */
let themeCache = { key: null, v: null };
export function themeColors() {
  const key = document.documentElement.dataset.theme || '';
  if (themeCache.key !== key) {
    const css = getComputedStyle(document.documentElement), token = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    themeCache = {
      key,
      v: {
        text: token('--text', '#ffffff'), dim: token('--dim', '#99a3b8'), faint: token('--faint', '#666'), hair: token('--hair', 'rgba(128,128,128,.2)'),
        fill: token('--fill', 'rgba(120,120,128,.38)'), fill2: token('--fill2', 'rgba(120,120,128,.26)'), accent: token('--accent', '#0a84ff'),
        good: token('--good', '#30d158'), bad: token('--bad', '#ff453a'), warn: token('--warn', '#ff9f0a'),
      },
    };
  }
  return themeCache.v;
}

/** Hex colour number -> "#rrggbb". */
export const hexCss = (n) => '#' + (n >>> 0).toString(16).padStart(6, '0').slice(-6);

/** Tiny event emitter. */
export class Emitter {
  constructor() { this._h = new Map(); }
  on(ev, fn) { (this._h.get(ev) || this._h.set(ev, new Set()).get(ev)).add(fn); return () => this.off(ev, fn); }
  off(ev, fn) { const s = this._h.get(ev); if (s) s.delete(fn); }
  emit(ev, ...args) { const s = this._h.get(ev); if (s) for (const fn of [...s]) fn(...args); }
}
