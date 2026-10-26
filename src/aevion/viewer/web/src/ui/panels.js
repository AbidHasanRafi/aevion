// Side drawer: layer toggles, display settings, and run information.

import { h, section, checkbox, select, slider, toast } from './dom.js';
import { QUALITY } from '../scene/stage.js';
import { COLOR_BY } from '../layers/trails.js';
import { fmtTime } from '../core/util.js';

const THEMES = [['dark', 'Dark'], ['light', 'Light'], ['print', 'Print']];
const QUALITIES = [['auto', 'Auto'], ['low', 'Low'], ['medium', 'Medium'], ['high', 'High']];
const COLOR_LABELS = { speed: 'Speed', altitude: 'Altitude', sinr: 'Link SINR', soc: 'Battery', none: 'Plain' };

export class SidePanel {
  constructor(ctx, root, layers) {
    this.ctx = ctx; this.root = root; this.layers = layers; this.sections = [];
    root.append(h('div', { class: 'drawer-title' }, 'Settings'));
    this.body = h('div', { class: 'drawer-body' }); root.append(this.body);
    this._layers(); this._display(); this._info();
  }

  _layers() {
    const groups = new Map();
    for (const L of this.layers) { if (!L.available) continue; (groups.get(L.uiGroup) || groups.set(L.uiGroup, []).get(L.uiGroup)).push(L); }
    const nodes = [];
    for (const [name, ls] of groups) nodes.push(h('div', { class: 'grp' }, h('div', { class: 'grp-title' }, name), ls.map((L) => checkbox(L.label, L.enabled, (v) => { L.enabled = v; }))));
    this.body.append(section('Layers', nodes));
    const cov = this.layers.find((L) => L.id === 'coverage' && L.available);
    if (cov) {
      const alts = cov.altitudes;
      this.body.append(section('Coverage map',
        select([['slice', 'One altitude'], ['volume', 'Stacked (volume)']], cov.mode, (v) => { this.ctx.settings.set('coverageMode', v); cov.onSetting('coverageMode', v); }),
        h('div', { class: 'row' }, h('span', { class: 'k' }, 'Altitude'), slider(Math.min(...alts), Math.max(...alts), alts.length > 1 ? (Math.max(...alts) - Math.min(...alts)) / (alts.length - 1) : 1, cov.current ?? alts[0], (v) => { this.ctx.settings.set('coverageAlt', v); cov.onSetting('coverageAlt', v); }))));
    }
  }

  _display() {
    const { ctx } = this, S = ctx.settings;
    const apply = (key, v) => { S.set(key, v); for (const L of this.layers) L.onSetting(key, v); ctx.events.emit('setting', key, v); };
    this.body.append(section('Display',
      h('div', { class: 'row' }, h('span', { class: 'k' }, 'Theme'), select(THEMES, S.get('theme'), (v) => apply('theme', v))),
      h('div', { class: 'row' }, h('span', { class: 'k' }, 'Quality'), select(QUALITIES, S.get('quality'), (v) => apply('quality', v))),
      h('div', { class: 'row' }, h('span', { class: 'k' }, 'Time of day'), slider(0, 24, 0.25, S.get('timeOfDay'), (v) => apply('timeOfDay', v))),
      h('div', { class: 'row' }, h('span', { class: 'k' }, 'Trail colour'), select(COLOR_BY.filter((c) => c !== 'sinr' || ctx.frames.overview.has('x:link_sinr_db')).map((c) => [c, COLOR_LABELS[c]]), S.get('trailColor'), (v) => apply('trailColor', v))),
      h('div', { class: 'row' }, h('span', { class: 'k' }, 'Trail length'), select([[0, 'Whole path'], [2, '2 s'], [5, '5 s'], [10, '10 s'], [30, '30 s']], S.get('trailSeconds'), (v) => apply('trailSeconds', +v))),
      h('div', { class: 'row' }, h('span', { class: 'k' }, 'Vehicle size'), slider(0.5, Math.max(12, Math.ceil(ctx.fleet.scale * 3)), 0.1, S.get('vehicleScale') || ctx.fleet.scale, (v) => apply('vehicleScale', v))),
      checkbox('Follow focused vehicle', S.get('followFocus'), (v) => apply('followFocus', v)),
      h('button', { onclick: () => { S.reset(); toast('Settings reset. Reload to apply everything'); } }, 'Reset settings')));
  }

  _info() {
    const { ctx } = this, m = ctx.meta, rows = [['Run', m.title], ['Vehicles', ctx.N], ['Duration', fmtTime(ctx.frames.overview.duration) + ' s'], ['Controller', m.controller], ['Trajectory', m.trajectory], ['Seed', m.seed], ['Backend', m.backend], ['Config hash', m.config_hash], ['Source', ctx.source.kind]].filter(([, v]) => v !== undefined && v !== '' && v !== null);
    this.body.append(section('About this run', h('table', { class: 'info' }, rows.map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', {}, String(v)))))));
  }

  add(title, ...nodes) { const s = section(title, ...nodes); this.body.append(s); return s; }
}

export { QUALITY };
