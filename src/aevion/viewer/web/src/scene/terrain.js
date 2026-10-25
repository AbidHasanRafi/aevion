// Ground: terrain from a heightmap (with optional imagery draped over it) or a flat checkered plane.

import * as THREE from 'three';
import { b64ToF32 } from '../core/util.js';

function checkerTexture(theme) {
  const c = document.createElement('canvas'); c.width = c.height = 256; const g = c.getContext('2d');
  const a = theme === 'print' ? '#e9e9e9' : '#5e8a57', b = theme === 'print' ? '#f6f6f6' : '#6a9763';
  g.fillStyle = a; g.fillRect(0, 0, 256, 256); g.fillStyle = b; g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
  g.strokeStyle = theme === 'print' ? 'rgba(0,0,0,.18)' : 'rgba(255,255,255,.16)'; g.lineWidth = 2; g.strokeRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; t.colorSpace = THREE.SRGBColorSpace; return t;
}

/** Bilinear terrain height function from a decoded heightmap description. */
export function heightFunction(hm, h) {
  const H = hm.rows, W = hm.cols;
  return (x, y) => {
    const fx = Math.min(Math.max((x - hm.origin[0]) / hm.res, 0), W - 1.001), fy = Math.min(Math.max((y - hm.origin[1]) / hm.res, 0), H - 1.001);
    const ix = Math.floor(fx), iy = Math.floor(fy), tx = fx - ix, ty = fy - iy;
    return (h[iy * W + ix] * (1 - tx) + h[iy * W + ix + 1] * tx) * (1 - ty) + (h[(iy + 1) * W + ix] * (1 - tx) + h[(iy + 1) * W + ix + 1] * tx) * ty;
  };
}

/**
 * Build the ground. Returns { height(x, y), object }, where `object` is added to the world group by the caller.
 * `imagery` may be { url, bounds: [xmin, ymin, xmax, ymax] } to drape a picture (satellite image, map, floor plan) over the ground.
 */
export function buildGround(worldData, { span, center, theme = 'dark', imagery = null } = {}) {
  const group = new THREE.Group();
  const gb = worldData ? worldData.bounds : [-50, -50, 0, 50, 50, 50];
  const gext = Math.max(span * 2.2, Math.max(gb[3] - gb[0], gb[4] - gb[1]) * 1.15, 60);
  const gcx = worldData ? (gb[0] + gb[3]) / 2 : center[0], gcy = worldData ? (gb[1] + gb[4]) / 2 : center[1];
  let height = () => 0;

  if (worldData && worldData.heightmap) {
    const hm = worldData.heightmap, H = hm.rows, W = hm.cols, h = b64ToF32(hm.h);
    const pos = new Float32Array(H * W * 3), col = new Float32Array(H * W * 3), uv = new Float32Array(H * W * 2);
    let hmax = 1e-6; for (const v of h) hmax = Math.max(hmax, v);
    const w = hm.res * (W - 1), d = hm.res * (H - 1);
    for (let iy = 0; iy < H; iy++) for (let ix = 0; ix < W; ix++) {
      const i = iy * W + ix, z = h[i], u = z / hmax;
      pos.set([hm.origin[0] + ix * hm.res, hm.origin[1] + iy * hm.res, z], i * 3);
      col.set([0.28 + 0.5 * u, 0.46 + 0.18 * u - 0.1 * u * u, 0.24 + 0.45 * u], i * 3);
      uv.set([ix / (W - 1), iy / (H - 1)], i * 2);
    }
    const idx = new (H * W > 65535 ? Uint32Array : Uint16Array)((H - 1) * (W - 1) * 6);
    let k = 0; for (let iy = 0; iy < H - 1; iy++) for (let ix = 0; ix < W - 1; ix++) { const a = iy * W + ix, b = a + 1, c = a + W, e = c + 1; idx.set([a, b, c, b, e, c], k); k += 6; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setIndex(new THREE.BufferAttribute(idx, 1)); g.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 });
    const m = new THREE.Mesh(g, mat); m.receiveShadow = true; group.add(m);
    height = heightFunction(hm, h);
    if (imagery) applyImagery(mat, imagery, { x0: hm.origin[0], y0: hm.origin[1], w, d }, () => { mat.vertexColors = false; mat.needsUpdate = true; });
  } else {
    const tex = checkerTexture(theme); tex.repeat.set(gext / 4, gext / 4);
    const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 1, metalness: 0 });
    const g = new THREE.Mesh(new THREE.PlaneGeometry(gext, gext), mat);
    g.position.set(gcx, gcy, -0.002); g.receiveShadow = true; group.add(g);
    if (imagery) {
      const [x0, y0, x1, y1] = imagery.bounds;
      applyImagery(mat, imagery, null, () => {
        mat.map = mat.userData.imagery; mat.needsUpdate = true;
        g.geometry.dispose(); g.geometry = new THREE.PlaneGeometry(x1 - x0, y1 - y0); g.position.set((x0 + x1) / 2, (y0 + y1) / 2, -0.002);
      });
    }
  }
  // an endless-looking apron: far beyond the detailed ground the fog turns it into the sky's horizon haze
  const outer = new THREE.Mesh(new THREE.PlaneGeometry(span * 80, span * 80), theme === 'print' ? new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }) : new THREE.MeshStandardMaterial({ color: 0x547c4f, roughness: 1, metalness: 0 }));
  outer.position.set(gcx, gcy, -0.06); group.add(outer);
  return { height, object: group, extent: gext };
}

function applyImagery(mat, imagery, _rect, done) {
  const loader = new THREE.TextureLoader();
  loader.load(imagery.url, (tex) => { tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8; mat.userData.imagery = tex; if (!mat.map || _rect) mat.map = tex; done(); },
    undefined, (err) => console.warn('ground imagery failed to load', err));
}
