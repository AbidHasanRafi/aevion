import test from 'node:test';
import assert from 'node:assert/strict';

import { Playback, SPEEDS } from '../src/core/playback.js';
import { OccupancyGrid } from '../src/data/occupancy.js';
import { EditorScene } from '../src/data/editor.js';

// ------------------------------------------------------------------------------------- playback
test('playback advances, loops and stops at the end', () => {
  let range = [0, 10];
  const p = new Playback(() => range, { speed: 2 });
  const events = []; p.on('loop', () => events.push('loop')); p.on('end', () => events.push('end'));
  p.tick(1); assert.equal(p.time, 2);
  p.tick(4); assert.equal(p.time, 0); assert.deepEqual(events, ['loop']); // 2 + 8 = 10 -> wrapped
  p.loop = false; p.seek(9); p.tick(1);
  assert.equal(p.time, 10); assert.equal(p.playing, false); assert.deepEqual(events, ['loop', 'end']);
  p.tick(5); assert.equal(p.time, 10); // paused: nothing moves
});

test('playback seek clamps and reports fractions', () => {
  const p = new Playback(() => [2, 12]);
  assert.equal(p.time, 2);
  assert.equal(p.seek(99), 12); assert.equal(p.seek(-5), 2);
  assert.equal(p.seekFraction(0.5), 7); assert.equal(p.fraction, 0.5); assert.equal(p.duration, 10);
  const empty = new Playback(() => [0, 0]); assert.equal(empty.fraction, 0); assert.doesNotThrow(() => empty.tick(1));
});

test('speed steps stay within the table', () => {
  const p = new Playback(() => [0, 1], { speed: 1 });
  p.faster(); assert.equal(p.speed, 2);
  for (let i = 0; i < 20; i++) p.faster();
  assert.equal(p.speed, SPEEDS[SPEEDS.length - 1]);
  for (let i = 0; i < 20; i++) p.slower();
  assert.equal(p.speed, SPEEDS[0]);
  p.setSpeed(1000); assert.equal(p.speed, SPEEDS[SPEEDS.length - 1]);
});

test('stepping frames pauses', () => {
  const p = new Playback(() => [0, 10]);
  p.stepFrames(3, 0.5); assert.equal(p.playing, false); assert.equal(p.time, 1.5);
  p.stepFrames(-100, 0.5); assert.equal(p.time, 0);
});

test('live playback never runs past the received data and can follow the edge', () => {
  let t1 = 5;
  const p = new Playback(() => [0, t1]); p.live = true;
  p.seek(4); p.tick(10); assert.equal(p.time, 5); assert.equal(p.playing, true);
  p.followLive = true; t1 = 8; p.tick(0.01); assert.equal(p.time, 8);
});

// --------------------------------------------------------------------------------------- occupancy
test('occupancy voxelises points, counts hits and respects the cap', () => {
  const g = new OccupancyGrid(0.5, 3);
  assert.ok(g.add(0.1, 0.1, 0.1)); assert.equal(g.add(0.4, 0.4, 0.4), false); // same voxel
  assert.ok(g.add(0.6, 0.1, 0.1)); assert.ok(g.add(-0.1, -0.1, -0.1)); // negative coordinates
  assert.equal(g.add(5, 5, 5), false); assert.equal(g.size, 3); // full
  assert.equal(g.add(NaN, 0, 0), false);
  const e = g.export(); assert.equal(e.count, 3);
  const c0 = [...e.centers.slice(0, 3)]; assert.deepEqual(c0, [0.25, 0.25, 0.25]); assert.equal(e.hits[0], 2);
  assert.equal(g.export(2).count, 1);
  assert.ok(g.has(0.2, 0.2, 0.3) && !g.has(3, 3, 3));
  const v = g.version; g.clear(); assert.equal(g.size, 0); assert.ok(g.version > v);
});

test('occupancy keys are collision free across axes', () => {
  const g = new OccupancyGrid(1, 1e6); let added = 0;
  for (let x = -3; x <= 3; x++) for (let y = -3; y <= 3; y++) for (let z = -3; z <= 3; z++) if (g.add(x + 0.5, y + 0.5, z + 0.5)) added++;
  assert.equal(added, 343); assert.equal(g.size, 343);
  assert.equal(g.add(1e9, 0, 0), false); assert.equal(g.add(0, -1e9, 0), false); // outside the addressable range
});

// ------------------------------------------------------------------------------------------ editor
test('editor scene add/update/remove and one start', () => {
  const s = new EditorScene();
  const b = s.add('box', 10, 20); assert.deepEqual(b.size, [10, 10, 12]);
  s.add('start', 0, 0); const st2 = s.add('start', 5, 5);
  assert.equal(s.of('start').length, 1); assert.equal(s.of('start')[0].id, st2.id);
  s.move(b.id, 11, 21); assert.deepEqual([s.get(b.id).x, s.get(b.id).y], [11, 21]);
  assert.equal(s.update('missing', { x: 1 }), null);
  assert.ok(s.remove(b.id)); assert.equal(s.remove(b.id), false);
  assert.throws(() => s.add('torus', 0, 0), /unknown object type/);
  assert.throws(() => s.add('box', NaN, 0), /finite/);
});

test('defaults are not shared between objects', () => {
  const s = new EditorScene(); const a = s.add('box', 0, 0), b = s.add('box', 5, 5);
  a.size[0] = 99; assert.equal(b.size[0], 10);
});

test('editor validation', () => {
  const s = new EditorScene();
  assert.deepEqual(s.validate(), ['place a start position']);
  s.add('start', 0, 0); assert.deepEqual(s.validate(), []);
  const box = s.add('box', 0, 0, { size: [4, 4, 5] });
  assert.deepEqual(s.validate(), ['the start position is inside a box']);
  s.update(box.id, { size: [4, -1, 5] }); assert.ok(s.validate().some((m) => m.includes('non-positive size')));
  s.remove(box.id); s.add('cylinder', 0.1, 0, { radius: 0.5, height: 8 });
  assert.deepEqual(s.validate(), ['the start position is inside a cylinder']);
});

test('editor exports the Aevion world and configuration', () => {
  const s = new EditorScene();
  s.add('start', 0, 0); s.add('box', 20, 0, { size: [10, 8, 6], z0: 0 }); s.add('cylinder', -10, 5); s.add('sphere', 3, 3, { z: 4, radius: 1 });
  s.add('waypoint', 30, 10, { z: 6 }); s.add('waypoint', 40, -5, { z: 7 }); s.add('cell', -50, 0, { z: 30, power_dbm: 40 });
  const w = s.toWorld();
  assert.deepEqual(w.boxes, [[15, -4, 0, 25, 4, 6]]); assert.deepEqual(w.cylinders, [[-10, 5, 0, 8, 0.6]]); assert.deepEqual(w.spheres, [[3, 3, 4, 1]]);
  assert.equal(w.box_colors.length, 1); assert.equal(w.bounds.length, 6);
  const c = s.toConfig({ name: 'demo', duration: 30, speed: 3 });
  assert.equal(c.schema_version, 1); assert.equal(c.name, 'demo'); assert.equal(c.world.type, 'from_dict');
  assert.deepEqual(c.vehicle.initial.pos, [0, 0, 1]);
  assert.deepEqual(c.trajectory, { type: 'waypoints', params: { points: [[0, 0, 1], [30, 10, 6], [40, -5, 7]], speed: 3 } });
  assert.deepEqual(c.network.params.sites, [{ pos: [-50, 0, 30], sectors: 1, tx_power_dbm: 40 }]);
  const empty = new EditorScene().toConfig(); assert.equal(empty.trajectory, undefined); assert.equal(empty.network, undefined);
});

test('editor JSON round trip and rejection of foreign files', () => {
  const s = new EditorScene(); s.add('start', 1, 2); s.add('box', 5, 5, { size: [3, 3, 3] });
  const back = EditorScene.fromJSON(JSON.parse(JSON.stringify(s.toJSON())));
  assert.deepEqual(back.toWorld(), s.toWorld()); assert.deepEqual(back.of('start')[0].x, 1);
  assert.throws(() => EditorScene.fromJSON({ format: 'other' }), /not an Aevion editor/);
  assert.throws(() => EditorScene.fromJSON(null), /not an Aevion editor/);
});
