// Layer registry: the order here is the drawing order and the order of the layer list in the UI.

import { TrailsLayer } from './trails.js';
import { ReferenceLayer, PlannedLayer } from './reference.js';
import { VectorsLayer, WashLayer } from './vectors.js';
import { LidarLayer, OccupancyLayer } from './lidar.js';
import { WirelessLayer } from './wireless.js';
import { BeamsLayer } from './beams.js';
import { CoverageLayer } from './coverage.js';
import { FrustumsLayer } from './frustums.js';
import { GhostsLayer } from './ghosts.js';
import { GeofencesLayer } from './geofences.js';
import { ActorsLayer } from './actors.js';

export const LAYER_CLASSES = [TrailsLayer, ReferenceLayer, PlannedLayer, GhostsLayer, VectorsLayer, WashLayer, LidarLayer, OccupancyLayer, FrustumsLayer, WirelessLayer, BeamsLayer, CoverageLayer, GeofencesLayer, ActorsLayer];

/** Instantiate every layer; a layer that fails to build is reported and skipped so one broken overlay never blanks the viewer. */
export function buildLayers(ctx, onError = console.warn) {
  const layers = [];
  for (const L of LAYER_CLASSES) {
    try { layers.push(new L(ctx)); } catch (e) { onError(`layer ${L.name} failed: ${e && e.message ? e.message : e}`); }
  }
  return layers;
}
