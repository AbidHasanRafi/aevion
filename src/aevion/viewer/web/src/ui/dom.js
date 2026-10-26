// Minimal DOM helpers: element factory, delegated events, and a toast. (No framework: the viewer ships as plain modules.)

import { icon } from './icons.js';

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** h('div', {class: 'a', onclick: fn, dataset: {x: 1}}, 'text', childElement) */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'html') el.innerHTML = v;
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}

let toastTimer = 0;
export function toast(msg, ms = 1500) {
  const t = document.getElementById('toast'); if (!t) return;
  t.textContent = msg; t.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

/** Icon-only button: the label is the accessible name and the tooltip (a glyph alone says nothing to a screen reader). */
export function iconButton(name, label, onclick, { class: cls = '', size, ...props } = {}) {
  return h('button', { type: 'button', class: `ibtn ${cls}`.trim(), title: label, 'aria-label': label, onclick, ...props }, icon(name, size ? { size } : {}));
}

/** Segmented control for a small set of related choices; `onpick(value)` is called on selection. Returns the container and a setter. */
export function segmented(options, value, onpick, label = '') {
  const btns = options.map(([v, text]) => h('button', { type: 'button', 'data-value': v, 'aria-pressed': String(v === value), class: v === value ? 'on' : '', onclick: () => onpick(v) }, text));
  const el = h('div', { class: 'seg', role: 'group', 'aria-label': label }, btns);
  const set = (val) => btns.forEach((b) => { const on = b.dataset.value === val; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
  return { el, set, btns };
}

/** A labelled switch; the row reads label-left, switch-right. */
export function checkbox(label, checked, onchange, { title = '' } = {}) {
  const input = h('input', { type: 'checkbox', role: 'switch', onchange: (e) => onchange(e.target.checked) }); input.checked = !!checked;
  return h('label', { class: 'chk', title }, input, h('span', {}, label));
}

export function select(options, value, onchange) {
  const s = h('select', { onchange: (e) => onchange(e.target.value) }, options.map(([v, label]) => h('option', { value: v }, label)));
  s.value = String(value); return s;
}

export function slider(min, max, step, value, oninput) {
  const fill = (el) => el.style.setProperty('--p', `${max > min ? ((+el.value - min) / (max - min)) * 100 : 0}%`);   // filled part of the track
  const s = h('input', { type: 'range', min, max, step, oninput: (e) => { fill(s); oninput(+e.target.value); } }); s.value = value; fill(s); return s;
}

/** Collapsible section for the side drawer. */
export function section(title, ...children) {
  const body = h('div', { class: 'sec-body' }, children);
  const head = h('button', { type: 'button', class: 'sec-head', 'aria-expanded': 'true', onclick: () => { body.hidden = !body.hidden; head.classList.toggle('closed', body.hidden); head.setAttribute('aria-expanded', String(!body.hidden)); } }, title);
  return h('div', { class: 'sec' }, head, body);
}
