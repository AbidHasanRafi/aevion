// Interface icons. One consistent set drawn on a 24-unit grid: 1.75 stroke, round caps and joins, so every glyph carries the same optical
// weight as the 13 px text next to it. Filled glyphs (play, pause, record) are used only for states, outlines everywhere else.

const NS = 'http://www.w3.org/2000/svg';

// [kind, geometry] pairs; kind 's' = stroked path data, 'f' = filled path data, 'c' = stroked circle "cx cy r", 'r' = stroked rect "x y w h rx"
const GLYPHS = {
  play: [['f', 'M7.6 5.3v13.4c0 .9 1 1.4 1.7.9l9.9-6.7c.6-.4.6-1.4 0-1.8L9.3 4.4c-.7-.5-1.7 0-1.7.9z']],
  pause: [['r', '6 4.6 4 14.8 1.3', 'f'], ['r', '14 4.6 4 14.8 1.3', 'f']],
  restart: [['s', 'M6.2 5v14'], ['f', 'M18.2 5.7v12.6c0 .8-.9 1.3-1.6.8l-8-6.3c-.5-.4-.5-1.2 0-1.6l8-6.3c.7-.5 1.6 0 1.6.8z']],
  loop: [['s', 'M16.5 3.5 20 7l-3.5 3.5'], ['s', 'M4 12.2V11a4 4 0 0 1 4-4h12'], ['s', 'M7.5 20.5 4 17l3.5-3.5'], ['s', 'M20 11.8V13a4 4 0 0 1-4 4H4']],
  camera: [['s', 'M3.8 9a2.6 2.6 0 0 1 2.6-2.6h1.1l.9-1.4a1.3 1.3 0 0 1 1.1-.6h5a1.3 1.3 0 0 1 1.1.6l.9 1.4h1.1A2.6 2.6 0 0 1 20.2 9v7.4a2.6 2.6 0 0 1-2.6 2.6H6.4a2.6 2.6 0 0 1-2.6-2.6z'], ['c', '12 12.7 3.3']],
  sidebar: [['r', '3.5 5 17 14 3.4'], ['s', 'M14.6 5v14']],
  xmark: [['s', 'M7 7l10 10'], ['s', 'M17 7 7 17']],
  chevronDown: [['s', 'm6.5 9.5 5.5 5.5 5.5-5.5']],
  chevronRight: [['s', 'm9.5 6.5 5.5 5.5-5.5 5.5']],
  plus: [['s', 'M12 5.5v13'], ['s', 'M5.5 12h13']],
  keyboard: [['r', '3 6.5 18 11 3'], ['s', 'M7 10.5h.01M10.5 10.5h.01M14 10.5h.01M17.5 10.5h.01M7.5 14.5h9']],
};

/** An inline <svg> for `name`. Size is in px; colour follows `currentColor`. */
export function icon(name, { size = 17 } = {}) {
  const parts = GLYPHS[name];
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('width', size); svg.setAttribute('height', size);
  svg.setAttribute('fill', 'none'); svg.setAttribute('stroke', 'currentColor'); svg.setAttribute('stroke-width', '1.75');
  svg.setAttribute('stroke-linecap', 'round'); svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true'); svg.setAttribute('focusable', 'false'); svg.classList.add('icon');
  for (const [kind, g, fill] of parts || []) {
    let el;
    if (kind === 'c') { const [cx, cy, r] = g.split(' '); el = document.createElementNS(NS, 'circle'); el.setAttribute('cx', cx); el.setAttribute('cy', cy); el.setAttribute('r', r); }
    else if (kind === 'r') { const [x, y, w, hh, rx] = g.split(' '); el = document.createElementNS(NS, 'rect'); for (const [k, v] of Object.entries({ x, y, width: w, height: hh, rx })) el.setAttribute(k, v); }
    else el = document.createElementNS(NS, 'path'), el.setAttribute('d', g);
    if (kind === 'f' || fill === 'f') { el.setAttribute('fill', 'currentColor'); el.setAttribute('stroke', 'none'); }
    svg.append(el);
  }
  return svg;
}

export const ICON_NAMES = Object.keys(GLYPHS);
