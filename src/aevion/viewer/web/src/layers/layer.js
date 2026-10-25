// Base class of every overlay drawn on top of the scene (trails, links, LiDAR, coverage, ...).
//
// A layer owns a THREE.Group inside the ENU world group. The app calls `update(cursor, dt)` once per rendered frame while the layer is
// enabled, `onFrames()` when live data arrived, and `onSetting(key, value)` when a setting changed.

import * as THREE from 'three';

export class Layer {
  /**
   * @param {object} ctx shared context: { stage, source, frames, cursor, meta, N, M, spec, fleet, settings, playback, span, center }
   * @param {string} id stable identifier (also the settings key `layer.<id>`)
   * @param {{label?: string, group?: string, defaultOn?: boolean, available?: boolean}} opts
   */
  constructor(ctx, id, { label = id, group = 'Display', defaultOn = true, available = true } = {}) {
    this.ctx = ctx; this.id = id; this.label = label; this.uiGroup = group;
    this.available = available;
    this.group = new THREE.Group(); this.group.name = `layer:${id}`;
    ctx.stage.W.add(this.group);
    const saved = ctx.settings?.get(`layer.${id}`);
    this._enabled = saved === undefined ? defaultOn : !!saved;
    this.group.visible = this._enabled && available;
  }

  get enabled() { return this._enabled; }
  set enabled(v) { this._enabled = !!v; this.group.visible = this._enabled && this.available; this.ctx.settings?.set(`layer.${this.id}`, this._enabled); this.onToggle?.(this._enabled); }

  /** Called every frame while enabled and available. */
  update(/* cursor, dt */) {}
  onFrames() {}
  onSetting(/* key, value */) {}
  onEvent(/* event */) {}

  dispose() {
    this.ctx.stage.W.remove(this.group);
    this.group.traverse((o) => { o.geometry?.dispose?.(); const m = o.material; if (m) (Array.isArray(m) ? m : [m]).forEach((x) => { x.map?.dispose?.(); x.dispose?.(); }); });
  }
}

/** Shared helpers for layers. */
export const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);

/** Text label as a sprite (canvas texture). Cheap enough for dozens of labels. */
export function textSprite(text, { color = '#ffffff', size = 1, bg = 'rgba(10,16,32,.7)' } = {}) {
  const c = document.createElement('canvas'), g = c.getContext('2d'), font = '600 28px system-ui,sans-serif';
  g.font = font; const w = Math.ceil(g.measureText(text).width) + 20; c.width = w; c.height = 44;
  g.font = font; g.fillStyle = bg; g.beginPath(); g.roundRect?.(0, 0, w, 44, 10); g.fill(); g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 10, 23);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set((w / 44) * size, size, 1); sp.renderOrder = 10;
  return sp;
}
