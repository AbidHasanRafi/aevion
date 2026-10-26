// In-browser scene editor: place buildings, trees, spheres, waypoints, base stations and a start position on the ground, then export an
// Aevion configuration (current schema) that `aevion run` executes. The data model lives in data/editor.js (pure, unit-tested).

import * as THREE from 'three';
import { h, toast } from './dom.js';
import { EditorScene } from '../data/editor.js';

const TOOLS = [['select', 'Select / move'], ['box', 'Building'], ['cylinder', 'Tree'], ['sphere', 'Sphere'], ['waypoint', 'Waypoint'], ['cell', 'Base station'], ['start', 'Start']];
const FIELDS = { box: [['size.0', 'width x'], ['size.1', 'depth y'], ['size.2', 'height'], ['z0', 'base z']], cylinder: [['radius', 'radius'], ['height', 'height']], sphere: [['radius', 'radius'], ['z', 'centre z']], waypoint: [['z', 'altitude']], cell: [['z', 'mast height'], ['power_dbm', 'power dBm']], start: [['z', 'altitude']] };

function buildMesh(o) {
  let m;
  if (o.type === 'box') { m = new THREE.Mesh(new THREE.BoxGeometry(o.size[0], o.size[1], o.size[2]), new THREE.MeshStandardMaterial({ color: 0xc9ccd4, roughness: 0.8 })); m.position.set(o.x, o.y, o.z0 + o.size[2] / 2); }
  else if (o.type === 'cylinder') { m = new THREE.Mesh(new THREE.CylinderGeometry(o.radius, o.radius * 1.08, o.height, 16), new THREE.MeshStandardMaterial({ color: 0x6b8e4e })); m.rotation.x = Math.PI / 2; m.position.set(o.x, o.y, o.height / 2); }
  else if (o.type === 'sphere') { m = new THREE.Mesh(new THREE.SphereGeometry(o.radius, 20, 14), new THREE.MeshStandardMaterial({ color: 0xcc8844 })); m.position.set(o.x, o.y, o.z ?? o.radius); }
  else if (o.type === 'waypoint') { m = new THREE.Mesh(new THREE.SphereGeometry(0.8, 14, 10), new THREE.MeshStandardMaterial({ color: 0xffd23f, emissive: 0x553d00 })); m.position.set(o.x, o.y, o.z); }
  else if (o.type === 'cell') { m = new THREE.Mesh(new THREE.ConeGeometry(2.2, 5, 3), new THREE.MeshStandardMaterial({ color: 0xff3a5c, emissive: 0x550011 })); m.rotation.x = Math.PI / 2; m.position.set(o.x, o.y, o.z); }
  else { m = new THREE.Mesh(new THREE.OctahedronGeometry(1, 0), new THREE.MeshStandardMaterial({ color: 0x39d98a, emissive: 0x0a3a22 })); m.position.set(o.x, o.y, o.z); }
  m.castShadow = true; m.userData.editorId = o.id; return m;
}

export class SceneEditor {
  constructor(ctx, panel) {
    this.ctx = ctx; this.scene = new EditorScene(); this.tool = 'select'; this.selected = null; this.meshes = new Map(); this.active = false; this.dragging = false;
    this.group = new THREE.Group(); this.group.name = 'editor'; ctx.stage.W.add(this.group);
    this.raycaster = new THREE.Raycaster(); this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.cursorMark = new THREE.Mesh(new THREE.RingGeometry(0.7, 1, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, side: THREE.DoubleSide })); this.cursorMark.visible = false; this.group.add(this.cursorMark);
    this.ui = this._build(); this.section = panel.add('Scene editor', this.ui);
    this._bind(); this.setActive(false);
  }

  _build() {
    this.toolBtns = TOOLS.map(([t, label]) => h('button', { 'data-tool': t, onclick: () => this.setTool(t), class: t === 'select' ? 'on' : '' }, label));
    this.list = h('div', { class: 'ed-list' }); this.props = h('div', { class: 'ed-props' }); this.issues = h('div', { class: 'ed-issues' });
    this.activeBtn = h('button', { onclick: () => this.setActive(!this.active) }, 'Start editing');
    const file = h('input', { type: 'file', accept: '.json', hidden: true, onchange: (e) => this._import(e.target.files[0]) });
    return h('div', { class: 'editor' },
      h('div', { class: 'hint' }, 'Click the ground to place the selected object; drag objects to move them; Delete removes the selection.'),
      this.activeBtn, h('div', { class: 'tools' }, this.toolBtns), this.list, this.props, this.issues,
      h('div', { class: 'ed-actions' },
        h('button', { onclick: () => this._download() }, 'Download config (.json)'), h('button', { onclick: () => this._copy() }, 'Copy config'),
        h('button', { onclick: () => this._saveScene() }, 'Save scene'), h('button', { onclick: () => file.click() }, 'Load scene…'), file,
        h('button', { class: 'danger', onclick: () => { this.scene.clear(); this.select(null); this.refresh(); } }, 'Clear')),
      h('div', { class: 'hint' }, 'Run it with:  aevion run scene.json'));
  }

  setActive(on) {
    this.active = on; this.activeBtn.textContent = on ? 'Stop editing' : 'Start editing'; this.group.visible = true;
    this.ctx.editorActive = on; if (on) this.ctx.playback.setPlaying(false);
    this.cursorMark.visible = false; this.ctx.events.emit('editor', on);
  }

  setTool(t) { this.tool = t; this.toolBtns.forEach((b) => b.classList.toggle('on', b.dataset.tool === t)); }

  _bind() {
    const cv = this.ctx.stage.canvas;
    const hit = (e) => {
      const r = cv.getBoundingClientRect(), ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.raycaster.setFromCamera(ndc, this.ctx.stage.camera);
      const p = new THREE.Vector3(); if (!this.raycaster.ray.intersectPlane(this.plane, p)) return null;
      const l = this.ctx.stage.W.worldToLocal(p); return { x: Math.round(l.x), y: Math.round(l.y), ndc };
    };
    cv.addEventListener('pointerdown', (e) => {
      if (!this.active || e.button !== 0) return;
      const pick = this._pick(e), p = hit(e);
      if (this.tool === 'select') { this.select(pick); if (pick) { this.dragging = true; this.ctx.stage.controls.enabled = false; cv.setPointerCapture(e.pointerId); } }
      else if (p) { const o = this.scene.add(this.tool, p.x, p.y); this.select(o.id); this.refresh(); }
    });
    cv.addEventListener('pointermove', (e) => {
      if (!this.active) return;
      const p = hit(e);
      if (p) { this.cursorMark.visible = this.tool !== 'select'; this.cursorMark.position.set(p.x, p.y, 0.05); }
      if (this.dragging && this.selected && p) { this.scene.move(this.selected, p.x, p.y); this.refresh(false); }
    });
    cv.addEventListener('pointerup', () => { if (this.dragging) { this.dragging = false; this.ctx.stage.controls.enabled = true; this.refresh(); } });
    addEventListener('keydown', (e) => { if (this.active && this.selected && (e.key === 'Delete' || e.key === 'Backspace') && !/INPUT|SELECT/.test(e.target.tagName)) { this.scene.remove(this.selected); this.select(null); this.refresh(); } });
  }

  _pick(e) {
    const cv = this.ctx.stage.canvas, r = cv.getBoundingClientRect();
    this.raycaster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.ctx.stage.camera);
    const hits = this.raycaster.intersectObjects([...this.meshes.values()], false);
    return hits.length ? hits[0].object.userData.editorId : null;
  }

  select(id) { this.selected = id; this._renderProps(); this.meshes.forEach((m, k) => { m.material.emissiveIntensity = k === id ? 1 : 0; if (k === id && !m.material.emissive.getHex()) m.material.emissive.setHex(0x335577); else if (k !== id && m.userData.base !== undefined) m.material.emissive.setHex(m.userData.base); }); }

  refresh(full = true) {
    for (const o of this.scene.objects) {
      let m = this.meshes.get(o.id);
      if (!m || full) { if (m) { this.group.remove(m); m.geometry.dispose(); } m = buildMesh(o); m.userData.base = m.material.emissive.getHex(); this.meshes.set(o.id, m); this.group.add(m); }
      else { m.position.x = o.x; m.position.y = o.y; }
    }
    for (const [id, m] of [...this.meshes]) if (!this.scene.get(id)) { this.group.remove(m); this.meshes.delete(id); }
    this._renderList(); this._renderProps(); this.issues.textContent = this.scene.validate().join(' · ');
  }

  _renderList() {
    this.list.replaceChildren(...this.scene.objects.map((o) => h('div', { class: 'ed-item' + (o.id === this.selected ? ' sel' : ''), onclick: () => this.select(o.id) }, `${o.type} (${o.x}, ${o.y})`)));
  }

  _renderProps() {
    this.props.replaceChildren(); const o = this.scene.get(this.selected); if (!o) return;
    for (const [path, label] of FIELDS[o.type] || []) {
      const [k, i] = path.split('.'), get = () => (i !== undefined ? o[k][+i] : o[k]);
      const input = h('input', { type: 'number', step: 'any', value: get(), onchange: (e) => { const v = parseFloat(e.target.value); if (!Number.isFinite(v)) return; if (i !== undefined) { const a = [...o[k]]; a[+i] = v; this.scene.update(o.id, { [k]: a }); } else this.scene.update(o.id, { [k]: v }); this.refresh(); } });
      this.props.append(h('label', {}, label, input));
    }
    this.props.append(h('button', { type: 'button', class: 'danger', onclick: () => { this.scene.remove(o.id); this.select(null); this.refresh(); } }, 'Delete'));
  }

  config() { return this.scene.toConfig({ name: this.ctx.meta.title || 'editor_scene' }); }
  _text() { return JSON.stringify(this.config(), null, 2); }
  _copy() { const t = this._text(); (navigator.clipboard?.writeText(t) || Promise.reject()).then(() => toast('Configuration copied'), () => toast('Copy failed: use Download')); }
  _download() { this._file('scene.json', this._text()); }
  _saveScene() { this._file('scene.editor.json', JSON.stringify(this.scene.toJSON(), null, 2)); }
  _file(name, text) { const a = document.createElement('a'); a.download = name; a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' })); a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
  async _import(file) {
    if (!file) return;
    try { this.scene = EditorScene.fromJSON(JSON.parse(await file.text())); this.meshes.forEach((m) => this.group.remove(m)); this.meshes.clear(); this.refresh(); toast(`Loaded ${this.scene.objects.length} objects`); }
    catch (e) { toast(`Cannot load: ${e.message}`, 3000); }
  }
}
