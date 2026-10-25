// Labels, units and display scaling of the channels the viewer knows about (pure).

const INFO = {
  speed: { label: 'Speed', unit: 'm/s' },
  altitude: { label: 'Altitude', unit: 'm' },
  err: { label: 'Tracking error', unit: 'cm', scale: 100 },
  power: { label: 'Electrical power', unit: 'W' },
  soc: { label: 'Battery', unit: '%', scale: 100 },
  thrust: { label: 'Total thrust', unit: 'N' },
  link_sinr_db: { label: 'SINR', unit: 'dB' },
  link_sinr_eff_db: { label: 'Effective SINR', unit: 'dB' },
  link_rate_mbps: { label: 'Link goodput', unit: 'Mbit/s' },
  link_rsrp_dbm: { label: 'RSRP', unit: 'dBm' },
  link_path_loss_db: { label: 'Path loss', unit: 'dB' },
  link_fading_db: { label: 'Fast fading', unit: 'dB' },
  link_doppler_hz: { label: 'Doppler', unit: 'Hz' },
  link_mcs: { label: 'MCS index', unit: '' },
  link_bler: { label: 'BLER', unit: '%', scale: 100 },
  link_share: { label: 'Scheduler share', unit: '%', scale: 100 },
  link_latency_ms: { label: 'HARQ latency', unit: 'ms' },
  link_interference_dbm: { label: 'Interference', unit: 'dBm' },
  link_n_handovers: { label: 'Handovers (cumulative)', unit: '' },
  link_n_rlf: { label: 'Radio-link failures', unit: '' },
  link_serving: { label: 'Serving cell', unit: '' },
  link_outage: { label: 'Outage', unit: '' },
  link_hops: { label: 'Mesh hops to gateway', unit: '' },
};

const PREFIX = { link_: 'Link' };

/** Display info for any channel name (`x:` prefix = extra channel recorded by the run). */
export function channelInfo(name) {
  const bare = name.startsWith('x:') ? name.slice(2) : name;
  if (INFO[bare]) return { scale: 1, ...INFO[bare] };
  let label = bare.replace(/_/g, ' ');
  for (const [p, nice] of Object.entries(PREFIX)) if (bare.startsWith(p)) label = `${nice} ${bare.slice(p.length).replace(/_/g, ' ')}`;
  return { label: label.charAt(0).toUpperCase() + label.slice(1), unit: '', scale: 1 };
}

/** Channels offered in the chart picker, in a stable order (core first, then extras). */
export function chartCandidates(store) {
  const core = ['speed', 'altitude', 'err', 'power', 'soc'].filter((n) => n === 'speed' || n === 'altitude' || store.has(n === 'err' ? 'err' : n));
  const extras = store.names().filter((n) => n.startsWith('x:')).sort();
  return [...core, ...extras];
}
