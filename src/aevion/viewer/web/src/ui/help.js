// Keyboard shortcuts overlay and the global key handler.

import { h } from './dom.js';

export const SHORTCUTS = [
  ['Space', 'play / pause'], ['← / →', 'step one frame (Shift: ten)'], ['[ / ]', 'slower / faster'], ['R', 'restart'],
  ['1 2 3 4', 'orbit / chase / top / FPV camera'], ['F', 'follow the focused vehicle'], ['Tab', 'next vehicle'],
  ['T', 'trails'], ['V', 'vectors'], ['L', 'LiDAR'], ['W', 'wireless links'], ['C', 'coverage map'],
  ['H', 'hide the interface'], ['D', 'side drawer'], ['S', 'screenshot'], ['?', 'this help'],
];

export function buildHelp() {
  const el = h('div', { id: 'help-overlay', hidden: true, onclick: () => { el.hidden = true; } },
    h('div', { class: 'help-card' }, h('h2', {}, 'Keyboard shortcuts'), h('table', {}, SHORTCUTS.map(([k, v]) => h('tr', {}, h('td', {}, h('kbd', {}, k)), h('td', {}, v)))), h('p', { class: 'dim' }, 'Click anywhere to close. Deep links: #t=5.5&cam=chase&play=0&ui=0')));
  document.body.append(el);
  return el;
}

/** Wire keys to actions: `a` is the object of callbacks provided by the app. */
export function bindKeys(a) {
  addEventListener('keydown', (e) => {
    const tag = e.target.tagName;
    if (tag === 'SELECT' || (tag === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase(), act = a.keys[k] || a.keys[e.key];
    if (act) { e.preventDefault(); act(e); }
  });
}
