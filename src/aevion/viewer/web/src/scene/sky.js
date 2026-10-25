// Procedural sky with a sun disc, horizon haze and a time of day. No textures: it works offline and costs one draw call.

import * as THREE from 'three';

const SKY_VERT = 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }';
const SKY_FRAG = `
varying vec3 vP;
uniform vec3 top; uniform vec3 horizon; uniform vec3 ground; uniform vec3 sunDir; uniform vec3 sunColor; uniform float sunSize; uniform float night; uniform float printMode;
void main(){
  float h = vP.y;
  vec3 c = mix(horizon, top, pow(clamp(h, 0.0, 1.0), 0.55));
  c = mix(c, ground, smoothstep(0.0, -0.25, h));
  float s = max(dot(normalize(vP), normalize(sunDir)), 0.0);
  c += sunColor * (pow(s, 900.0) * 6.0 * sunSize + pow(s, 12.0) * 0.18 * (1.0 - night));
  // sparse stars at night
  vec3 q = floor(normalize(vP) * 140.0); float r = fract(sin(dot(q, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
  c += vec3(step(0.9975, r)) * night * smoothstep(0.05, 0.4, h);
  gl_FragColor = vec4(mix(c, vec3(1.0), printMode), 1.0);
}`;

export function sunDirection(hour, out = new THREE.Vector3()) {
  // simple model: sunrise 6:00, noon 12:00 (elevation 62 degrees), sunset 18:00; azimuth sweeps east -> south -> west
  const a = ((hour - 6) / 12) * Math.PI;               // 0 at sunrise, pi at sunset
  const elev = Math.sin(a) * 62 * Math.PI / 180;
  const az = Math.PI * 0.5 + a * 0.85 - Math.PI * 0.43;
  out.set(Math.cos(elev) * Math.sin(az), Math.sin(elev), -Math.cos(elev) * Math.cos(az));
  return out.normalize();
}

export function makeSky() {
  const uniforms = {
    top: { value: new THREE.Color(0x3d78c9) }, horizon: { value: new THREE.Color(0xcfe1f2) }, ground: { value: new THREE.Color(0x44503a) },
    sunDir: { value: new THREE.Vector3(0, 1, 0) }, sunColor: { value: new THREE.Color(0xfff2d6) }, sunSize: { value: 1 }, night: { value: 0 }, printMode: { value: 0 },
  };
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(3000, 32, 16), new THREE.ShaderMaterial({ side: THREE.BackSide, depthWrite: false, fog: false, uniforms, vertexShader: SKY_VERT, fragmentShader: SKY_FRAG }));
  mesh.frustumCulled = false; mesh.renderOrder = -10;
  return { mesh, uniforms, setPrint(on) { uniforms.printMode.value = on ? 1 : 0; } };
}

const mix = (a, b, t) => new THREE.Color().copy(a).lerp(b, Math.min(Math.max(t, 0), 1));

/** Colours for a given hour: returns { night } (0 = day, 1 = night) and updates the stage's sky, fog and lights. */
export function applyTimeOfDay(hour, stage) {
  const dir = sunDirection(hour), elev = dir.y;                     // -1..1
  const day = Math.min(Math.max((elev + 0.08) / 0.4, 0), 1);        // 0 at night, 1 in daylight
  const dusk = Math.min(Math.max(1 - Math.abs(elev) / 0.25, 0), 1) * (elev > -0.12 ? 1 : 0); // orange glow near the horizon
  const night = 1 - day;
  const u = stage.sky.uniforms;
  const topDay = new THREE.Color(0x3d78c9), topNight = new THREE.Color(0x050a1a);
  const horDay = new THREE.Color(0xcfe1f2), horNight = new THREE.Color(0x0e1830), horDusk = new THREE.Color(0xff9b5e);
  u.top.value.copy(mix(topNight, topDay, day));
  u.horizon.value.copy(mix(mix(horNight, horDay, day), horDusk, dusk * 0.8));
  u.ground.value.copy(u.horizon.value).multiplyScalar(0.8);      // below the horizon the sky is the haze the ground fades into
  u.sunDir.value.copy(dir); u.night.value = night; u.sunSize.value = elev > -0.05 ? 1 : 0;
  u.sunColor.value.copy(mix(new THREE.Color(0xfff2d6), new THREE.Color(0xff8a3c), dusk));
  stage.hemi.intensity = 0.15 + 0.7 * day; stage.hemi.color.copy(mix(new THREE.Color(0x4a5a8a), new THREE.Color(0xdfeeff), day));
  stage.sun.intensity = 2.3 * Math.max(elev, 0) ** 0.6 * day + 0.0001;
  stage.sun.color.copy(mix(new THREE.Color(0xfff4e0), new THREE.Color(0xffa060), dusk));
  stage.renderer.toneMappingExposure = 0.8 + 0.3 * day;
  if (stage.scene.fog) stage.scene.fog.color.copy(u.horizon.value);
  return { night, day, dusk, elevation: elev };
}
