// Scene-editor model (pure): objects the user places in the browser and their conversion to an Aevion configuration.
//
// The exported object follows the Aevion configuration schema (a *fragment*: world, trajectory, network, vehicle), so it can be
// merged into any scenario or run directly: `aevion.viewer.editor.config_from_editor(json)` validates and builds it in Python.

export const TYPES = ['box', 'cylinder', 'sphere', 'waypoint', 'cell', 'start'];

const DEFAULTS = {
  box: { size: [10, 10, 12], z0: 0 },
  cylinder: { radius: 0.6, height: 8 },
  sphere: { radius: 2 },
  waypoint: { z: 5 },
  cell: { z: 25, power_dbm: 46 },
  start: { z: 1 },
};

let _id = 0;

export class EditorScene {
  constructor() { this.objects = []; this.version = 0; }

  add(type, x, y, props = {}) {
    if (!TYPES.includes(type)) throw new Error(`unknown object type '${type}'`);
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error('position must be finite');
    if (type === 'start') this.objects = this.objects.filter((o) => o.type !== 'start'); // exactly one start
    const o = { id: `o${++_id}`, type, x, y, ...JSON.parse(JSON.stringify(DEFAULTS[type])), ...props };
    this.objects.push(o);
    this.version++;
    return o;
  }

  remove(id) { const n = this.objects.length; this.objects = this.objects.filter((o) => o.id !== id); if (this.objects.length !== n) this.version++; return this.objects.length !== n; }
  get(id) { return this.objects.find((o) => o.id === id) || null; }
  update(id, props) { const o = this.get(id); if (!o) return null; Object.assign(o, props); this.version++; return o; }
  move(id, x, y) { return this.update(id, { x, y }); }
  clear() { this.objects = []; this.version++; }
  of(type) { return this.objects.filter((o) => o.type === type); }

  /** Problems that would make the exported configuration invalid or surprising. */
  validate() {
    const issues = [];
    if (!this.of('start').length) issues.push('place a start position');
    for (const o of this.objects) {
      if (o.type === 'box' && o.size.some((s) => !(s > 0))) issues.push(`box ${o.id} has a non-positive size`);
      if ((o.type === 'cylinder' || o.type === 'sphere') && !(o.radius > 0)) issues.push(`${o.type} ${o.id} has a non-positive radius`);
      if (o.type === 'cylinder' && !(o.height > 0)) issues.push(`cylinder ${o.id} has a non-positive height`);
    }
    const s = this.of('start')[0];
    if (s) {
      for (const o of this.objects) {
        if (o.type === 'box' && Math.abs(s.x - o.x) < o.size[0] / 2 && Math.abs(s.y - o.y) < o.size[1] / 2 && s.z < o.z0 + o.size[2]) issues.push('the start position is inside a box');
        if (o.type === 'cylinder' && Math.hypot(s.x - o.x, s.y - o.y) < o.radius && s.z < o.height) issues.push('the start position is inside a cylinder');
      }
    }
    return issues;
  }

  /** World dictionary in the format of `World.to_dict()` / `World.from_dict()`. */
  toWorld(bounds = null) {
    const boxes = this.of('box').map((o) => [o.x - o.size[0] / 2, o.y - o.size[1] / 2, o.z0, o.x + o.size[0] / 2, o.y + o.size[1] / 2, o.z0 + o.size[2]]);
    const cylinders = this.of('cylinder').map((o) => [o.x, o.y, 0, o.height, o.radius]);
    const spheres = this.of('sphere').map((o) => [o.x, o.y, o.z ?? o.radius, o.radius]);
    if (!bounds) {
      const xs = this.objects.map((o) => o.x), ys = this.objects.map((o) => o.y);
      const pad = 40;
      bounds = this.objects.length ? [Math.min(...xs) - pad, Math.min(...ys) - pad, 0, Math.max(...xs) + pad, Math.max(...ys) + pad, 100] : [-100, -100, 0, 100, 100, 100];
    }
    return { boxes, spheres, cylinders, bounds, box_colors: boxes.map(() => [0.75, 0.75, 0.78]), sphere_colors: spheres.map(() => [0.8, 0.55, 0.3]), cyl_colors: cylinders.map(() => [0.42, 0.56, 0.3]) };
  }

  /** Aevion configuration fragment (current schema). */
  toConfig({ name = 'editor_scene', duration = 20, speed = 2.0 } = {}) {
    const start = this.of('start')[0];
    const wps = this.of('waypoint');
    const cfg = { schema_version: 1, name, duration, world: { type: 'from_dict', params: { data: this.toWorld() } } };
    if (start) cfg.vehicle = { initial: { pos: [start.x, start.y, start.z] } };
    if (wps.length) cfg.trajectory = { type: 'waypoints', params: { points: [...(start ? [[start.x, start.y, start.z]] : []), ...wps.map((w) => [w.x, w.y, w.z])], speed } };
    const cells = this.of('cell');
    if (cells.length) cfg.network = { type: 'nr_cellular', rate_hz: 10, params: { sites: cells.map((c) => ({ pos: [c.x, c.y, c.z], sectors: 1, tx_power_dbm: c.power_dbm })), los_mode: 'geometric' } };
    return cfg;
  }

  toJSON() { return { format: 'aevion.editor/1', objects: this.objects }; }

  static fromJSON(d) {
    const s = new EditorScene();
    if (!d || d.format !== 'aevion.editor/1' || !Array.isArray(d.objects)) throw new Error('not an Aevion editor file');
    for (const o of d.objects) s.add(o.type, o.x, o.y, Object.fromEntries(Object.entries(o).filter(([k]) => !['id', 'type', 'x', 'y'].includes(k))));
    return s;
  }
}
