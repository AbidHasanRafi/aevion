// Sparse voxel occupancy accumulated from LiDAR hits (pure). Used by the occupancy layer.

export class OccupancyGrid {
  /**
   * @param {number} voxel edge length in metres
   * @param {number} maxVoxels hard cap on stored voxels (oldest are not evicted: new ones are ignored once full)
   */
  constructor(voxel = 0.5, maxVoxels = 60000) {
    this.voxel = voxel; this.maxVoxels = maxVoxels;
    this.cells = new Map();           // key -> {ix, iy, iz, hits}
    this.version = 0;
  }

  _key(ix, iy, iz) {
    // 17 bits per axis (51 bits in total: exact in a double). Indices must satisfy |i| < 65536 (+/- 32 km at 0.5 m voxels).
    return ((ix + 65536) * 131072 + (iy + 65536)) * 131072 + (iz + 65536);
  }

  add(x, y, z) {
    if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))) return false;
    const v = this.voxel, ix = Math.floor(x / v), iy = Math.floor(y / v), iz = Math.floor(z / v);
    if (Math.abs(ix) >= 65536 || Math.abs(iy) >= 65536 || Math.abs(iz) >= 65536) return false;
    const k = this._key(ix, iy, iz);
    const c = this.cells.get(k);
    if (c) { c.hits++; return false; }
    if (this.cells.size >= this.maxVoxels) return false;
    this.cells.set(k, { ix, iy, iz, hits: 1 });
    this.version++;
    return true;
  }

  /** Centre of every occupied voxel with its hit count, as flat arrays (for instanced rendering). */
  export(minHits = 1) {
    const out = [], hits = [], v = this.voxel;
    for (const c of this.cells.values()) if (c.hits >= minHits) { out.push((c.ix + 0.5) * v, (c.iy + 0.5) * v, (c.iz + 0.5) * v); hits.push(c.hits); }
    return { centers: Float32Array.from(out), hits: Float32Array.from(hits), count: hits.length };
  }

  get size() { return this.cells.size; }
  clear() { this.cells.clear(); this.version++; }
  has(x, y, z) { const v = this.voxel; return this.cells.has(this._key(Math.floor(x / v), Math.floor(y / v), Math.floor(z / v))); }
}
