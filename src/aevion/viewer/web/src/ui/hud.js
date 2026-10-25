// Heads-up display: title card and the live readout of the focused vehicle.

import { h } from './dom.js';
import { fmt } from '../core/util.js';
import { valueAt } from '../data/derived.js';

export class Hud {
  constructor(ctx, titleRoot, hudRoot) {
    this.ctx = ctx; const { meta } = ctx;
    titleRoot.append(h('h1', {}, meta.title || 'Aevion run'), this.sub = h('div', { class: 'sub' }));
    this.rows = {};
    const row = (key, label, extra = {}) => { const v = h('span', { class: 'v' }, '–'); const r = h('div', { class: 'row', ...extra }, h('span', { class: 'k' }, label), v); hudRoot.append(r); this.rows[key] = { row: r, v }; return r; };
    this.vsel = h('select', { onchange: (e) => ctx.fleet.setFocus(+e.target.value) }, Array.from({ length: ctx.N }, (_, i) => h('option', { value: i }, `#${i + 1}`)));
    this.vsel.disabled = ctx.N === 1;
    hudRoot.append(h('div', { class: 'row' }, h('span', { class: 'k' }, 'Vehicle'), h('span', { class: 'v' }, this.vsel)));
    row('pos', 'Position'); row('spd', 'Speed'); row('alt', 'Altitude'); row('att', 'Roll / Pitch / Yaw');
    if (ctx.frames.overview.has('err')) row('err', 'Tracking error');
    row('pwr', 'Power'); row('soc', 'Battery'); hudRoot.append(h('div', { class: 'bar' }, this.socBar = h('i')));
    if (ctx.frames.overview.has('x:link_sinr_db')) { row('link', 'Link SINR / rate'); row('cell', 'Serving cell'); }
    row('wind', 'Wind');
    this.rotors = h('div', { id: 'rotors' }); hudRoot.append(this.rotors);
    this.rotBars = [];
    for (let m = 0; m < ctx.M; m++) { const bar = h('i'), val = h('span', { class: 'rv' }); this.rotors.append(h('div', { class: 'rb' }, h('span', {}, `M${m + 1}`), h('div', { class: 'bar' }, bar), val)); this.rotBars.push([bar, val]); }
    this.setSub();
  }

  setSub(extra = '') {
    const { meta, N } = this.ctx, dur = this.ctx.frames.overview.duration || meta.duration || 0;
    this.sub.textContent = [meta.vehicle?.name, `${N} vehicle${N > 1 ? 's' : ''}`, meta.controller, meta.trajectory, `${dur.toFixed(1)} s`, extra].filter(Boolean).join(' · ');
  }

  update(cursor) {
    const { ctx } = this, v = ctx.fleet.focus, P = ctx.fleet.poses[v], R = this.rows, outage = ctx.source.layers?.wireless?.outage_db ?? -6;
    R.pos.v.textContent = `${fmt(P.p.x)}, ${fmt(P.p.y)}, ${fmt(P.p.z)} m`;
    R.spd.v.textContent = `${fmt(valueAt(cursor, 'speed', v))} m/s`;
    R.alt.v.textContent = `${fmt(P.p.z)} m`;
    R.att.v.textContent = [0, 1, 2].map((j) => `${(cursor.get('euler', v, j) * 180 / Math.PI).toFixed(0)}°`).join(' ');
    if (R.err) R.err.v.textContent = `${fmt(cursor.get('err', v, 0) * 100, 1)} cm`;
    R.pwr.v.textContent = `${fmt(cursor.get('power', v, 0), 0)} W`;
    const soc = cursor.get('soc', v, 0); R.soc.v.textContent = `${fmt(soc * 100, 1)} %`; this.socBar.style.width = `${Math.max(0, Math.min(100, soc * 100))}%`;
    R.wind.v.textContent = `${fmt(valueAt(cursor, 'wind_speed', v))} m/s`;
    if (R.link) {
      const s = cursor.get('x:link_sinr_db', v, 0), r = cursor.has('x:link_rate_mbps') ? cursor.get('x:link_rate_mbps', v, 0) : NaN, los = cursor.has('x:link_los') ? cursor.get('x:link_los', v, 0) > 0.5 : null;
      R.link.v.textContent = `${fmt(s, 1)} dB · ${fmt(r, 0)} Mb/s${los === null ? '' : los ? ' LoS' : ' NLoS'}`; R.link.v.style.color = s < outage ? 'var(--bad)' : '';
      const ho = cursor.has('x:link_n_handovers') ? ` · ${Math.round(cursor.get('x:link_n_handovers', v, 0))} HO` : '';
      R.cell.v.textContent = `${Math.round(cursor.get('x:link_serving', v, 0))}${ho}`;
    }
    for (let m = 0; m < this.rotBars.length; m++) { const w = cursor.get('rotor', v, m); this.rotBars[m][0].style.width = `${Math.min(100, (w / ctx.spec.w_max) * 100)}%`; this.rotBars[m][1].textContent = `${(w * 60 / (2 * Math.PI) / 1000).toFixed(1)}k`; }
  }
}
