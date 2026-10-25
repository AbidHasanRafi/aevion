// Pure helpers for drawing geofence zones.

/** Vertical extent [z0, z1] of a zone, with open ends replaced by the visible range. */
export function zoneExtent(zone, span) {
  const top = Math.max(span * 0.6, 40);
  const z0 = zone.z_min === null || zone.z_min === undefined ? 0 : zone.z_min;
  const z1 = zone.z_max === null || zone.z_max === undefined ? Math.max(top, z0 + 10) : zone.z_max;
  return [z0, z1];
}
