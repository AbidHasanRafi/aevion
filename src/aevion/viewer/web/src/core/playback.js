// Playback clock (pure): play/pause, speed, looping, scrubbing, and following the live edge of a growing run.

import { clamp, Emitter } from './util.js';

export const SPEEDS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16];

export class Playback extends Emitter {
  /**
   * @param {() => [number, number]} range function returning the [t0, t1] currently available
   */
  constructor(range, { playing = true, speed = 1, loop = true } = {}) {
    super();
    this.range = range; this.playing = playing; this.speed = speed; this.loop = loop;
    this.time = range()[0]; this.live = false; this.followLive = false;
  }

  get t0() { return this.range()[0]; }
  get t1() { return this.range()[1]; }
  get duration() { const [a, b] = this.range(); return Math.max(b - a, 0); }
  /** Position in [0, 1] along the available range. */
  get fraction() { const [a, b] = this.range(); return b > a ? (this.time - a) / (b - a) : 0; }

  seek(t) { const [a, b] = this.range(); this.time = clamp(t, a, b); this.emit('seek', this.time); return this.time; }
  seekFraction(f) { const [a, b] = this.range(); return this.seek(a + clamp(f, 0, 1) * (b - a)); }
  setPlaying(p) { if (this.playing !== p) { this.playing = p; this.emit('playing', p); } }
  toggle() { this.setPlaying(!this.playing); }
  setSpeed(s) { this.speed = clamp(s, SPEEDS[0], SPEEDS[SPEEDS.length - 1]); this.emit('speed', this.speed); }
  faster() { const i = SPEEDS.findIndex((s) => s >= this.speed - 1e-9); this.setSpeed(SPEEDS[Math.min(i + 1, SPEEDS.length - 1)]); }
  slower() { const i = SPEEDS.findIndex((s) => s >= this.speed - 1e-9); this.setSpeed(SPEEDS[Math.max(i - 1, 0)]); }

  /** Step by n recorded frames of duration dtRec (pauses playback). */
  stepFrames(n, dtRec) { this.setPlaying(false); return this.seek(this.time + n * dtRec); }

  /** Advance by a wall-clock interval (seconds). Returns the new time. */
  tick(dt) {
    const [a, b] = this.range();
    if (this.live && this.followLive) { this.time = b; return this.time; }
    if (!this.playing) return this.time;
    this.time += dt * this.speed;
    if (this.time >= b) {
      if (this.live) { this.time = b; }              // never run past what has been received
      else if (this.loop && b > a) { this.time = a; this.emit('loop'); }
      else { this.time = b; this.setPlaying(false); this.emit('end'); }
    }
    return this.time;
  }
}
