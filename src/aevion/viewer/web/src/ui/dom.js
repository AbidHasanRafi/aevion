// Minimal DOM helpers: element factory, delegated events, and a toast. (No framework: the viewer ships as plain modules.)

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
  t.textContent = msg; t.style.opacity = 1; clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.style.opacity = 0; }, ms);
}

export function checkbox(label, checked, onchange, { title = '' } = {}) {
  const input = h('input', { type: 'checkbox', onchange: (e) => onchange(e.target.checked) }); input.checked = !!checked;
  return h('label', { class: 'chk', title }, input, label);
}

export function select(options, value, onchange) {
  const s = h('select', { onchange: (e) => onchange(e.target.value) }, options.map(([v, label]) => h('option', { value: v }, label)));
  s.value = String(value); return s;
}

export function slider(min, max, step, value, oninput) {
  const s = h('input', { type: 'range', min, max, step, oninput: (e) => oninput(+e.target.value) }); s.value = value; return s;
}

/** Collapsible section for the side drawer. */
export function section(title, ...children) {
  const body = h('div', { class: 'sec-body' }, children);
  const head = h('button', { class: 'sec-head', onclick: () => { body.hidden = !body.hidden; head.classList.toggle('closed', body.hidden); } }, title);
  return h('div', { class: 'sec' }, head, body);
}
