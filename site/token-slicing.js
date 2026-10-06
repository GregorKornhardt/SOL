import { cloudFrame } from './cloud-geometry.js';

const states = new WeakMap();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const duration = 4200;
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { const t = clamp(x); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;
const move = (element, [x, y]) => { element.setAttribute('cx', x); element.setAttribute('cy', y); };
const segment = (element, a, b) => {
  for (const [key, value] of Object.entries({ x1: a[0], y1: a[1], x2: b[0], y2: b[1] })) element.setAttribute(key, value);
};

function updateControls(state) {
  state.element.dataset.slicingPhase = state.phase;
  state.toggle.textContent = {
    clouds: '▶ Project onto ξ', slicing: 'Ⅱ Pause slicing', paused: '▶ Continue slicing', rows: '↺ Replay slicing',
  }[state.phase];
  state.reset.hidden = state.phase === 'clouds';
  const step = state.elapsed < 650 ? 'direction' : state.elapsed < 2800 ? 'projection' : 'rows';
  const status = state.phase === 'clouds' ? '12 token clouds · one shared direction ξ'
    : state.phase === 'paused' ? 'Slicing paused.'
      : state.phase === 'rows' ? '12 distributions · one labeled row per sequence'
        : { direction: 'Use the same direction ξ for every sequence.', projection: 'Tokens move perpendicularly onto the shared line.', rows: 'Keep each sequence’s projected tokens in its own row.' }[step];
  if (state.status.textContent !== status) state.status.textContent = status;
}

// Row spacing for the 12 projected sequences; 25px when the plot is 430px tall.
const rowGap = height => Math.min(25, (height - 105) / 13);

function paint(state) {
  const { elapsed, width: w, height: h, direction } = state;
  if (!w || !h) return;
  const frame = cloudFrame(w, h);
  const project = ease((elapsed - 650) / 1500);
  const unfold = ease((elapsed - 2800) / 1400);
  const rowX = value => 48 + (value + 2) / 4 * (w - 68);
  const gap = rowGap(h);
  const rowY = row => 48 + row.sequence * gap + row.corpus * (6 * gap + 34);
  state.axes.style.opacity = 1 - unfold;
  state.axis.style.opacity = 1 - unfold;
  state.rowAxes.style.opacity = unfold;
  state.svg.querySelector('.slice-heading').textContent = unfold < .5 ? 'Shared direction ξ' : '⟨ξ, hᵢ⟩';
  const end = value => frame.point(direction.map(x => x * value));
  segment(state.axis, end(-2), end(2));
  state.svg.querySelector('.slice-arrow').style.opacity = 1 - unfold;
  const arrow = end(2);
  state.svg.querySelector('.slice-arrow').setAttribute('d', `M${arrow[0] + direction[0] * -9 + direction[1] * 4},${arrow[1] + direction[1] * 9 + direction[0] * 4}L${arrow[0]},${arrow[1]}L${arrow[0] + direction[0] * -9 - direction[1] * 4},${arrow[1] + direction[1] * 9 - direction[0] * 4}`);
  state.rows.forEach(row => {
    const starts = row.points.map(frame.point);
    const xs = starts.map(p => p[0]), ys = starts.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const rx = (Math.max(...xs) - Math.min(...xs)) / 2 + 10;
    const ry = (Math.max(...ys) - Math.min(...ys)) / 2 + 10;
    for (const [key, value] of Object.entries({ cx, cy, rx: rx * 1.15, ry: ry * 1.15 })) row.outline.setAttribute(key, value);
    row.outline.style.opacity = (1 - project) * .8;
    segment(row.axis, [42, rowY(row)], [w - 18, rowY(row)]);
    row.axis.style.opacity = unfold;
    row.label.setAttribute('x', mix(cx, 25, unfold));
    row.label.setAttribute('y', mix(cy - ry * 1.15 - 6, rowY(row) + 4, unfold));
    row.dots.forEach((dot, token) => {
      // Keep the token's identity while projecting; sorted coordinates define its distribution.
      const [x, y] = row.points[token];
      const value = x * direction[0] + y * direction[1];
      const target = end(value);
      const projected = starts[token].map((p, axis) => mix(p, target[axis], project));
      move(dot, [mix(projected[0], rowX(value), unfold), mix(projected[1], rowY(row), unfold)]);
      move(row.ghosts[token], starts[token]);
      row.ghosts[token].style.opacity = project * (1 - unfold) * .22;
      segment(row.guides[token], starts[token], target);
      row.guides[token].style.opacity = (row.active ? .45 : 0) * clamp(elapsed / 650) * (1 - unfold);
    });
  });
}

function measure(state) {
  if (states.get(state.element) !== state) return;
  // Layout pixels, so a CSS transform on the page scales text and dots too.
  const width = state.svg.clientWidth, height = state.svg.clientHeight;
  Object.assign(state, { width, height });
  state.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  state.axes.innerHTML = `<path d="M36 30V${height - 30}H${width - 14}"/><text x="${width - 19}" y="${height - 8}">x</text><text x="14" y="28">y</text>`;
  const rowX = value => 48 + (value + 2) / 4 * (width - 68);
  state.rowAxes.innerHTML = `<text x="12" y="24" class="corpus-0">REFERENCE</text><text x="12" y="${58 + 6 * rowGap(height)}" class="corpus-1">GENERATED</text><path d="M48 ${height - 30}H${width - 20}"/>${[-2, -1, 0, 1, 2].map(value => `<text x="${rowX(value)}" y="${height - 10}" text-anchor="middle">${value}</text>`).join('')}`;
  paint(state);
}

function stop(state) { cancelAnimationFrame(state.frame); state.frame = null; }
function finish(state) {
  stop(state); state.elapsed = duration; state.phase = 'rows';
  paint(state); updateControls(state);
}
function pause(state) {
  if (state.phase !== 'slicing') return;
  state.elapsed = Math.min(duration, performance.now() - state.started);
  stop(state); state.phase = 'paused'; paint(state); updateControls(state);
}
function play(state) {
  stop(state);
  if (state.phase !== 'paused') state.elapsed = 0;
  state.phase = 'slicing';
  if (reducedMotion.matches) { finish(state); return; }
  state.started = performance.now() - state.elapsed;
  paint(state); updateControls(state);
  const tick = now => {
    state.elapsed = Math.min(duration, now - state.started);
    paint(state); updateControls(state);
    if (state.elapsed >= duration) finish(state);
    else state.frame = requestAnimationFrame(tick);
  };
  state.frame = requestAnimationFrame(tick);
}

export function renderTokenSlicing(element, corpora, slices, traced) {
  let state = states.get(element);
  if (state && state.corpora !== corpora) { disposeTokenSlicing(element); state = null; }
  if (!state) {
    element.innerHTML = `<div class="embedding-controls"><button class="button button-small" data-slice-toggle>▶ Project onto ξ</button><button class="text-button" data-slice-reset hidden>Show clouds</button></div><p class="embedding-status slicing-status" role="status"></p><svg class="slicing-plot" xmlns="http://www.w3.org/2000/svg" role="group" aria-label="Project 12 token clouds onto the same direction ξ, then keep one labeled row per sequence"><g class="embedding-axes slice-axes"></g><text class="slice-heading" x="100%" y="24" text-anchor="end"></text><g class="slice-row-axes"></g><line class="slice-direction"/><path class="slice-arrow"/>${[corpora.reference, corpora.generated].map((clouds, corpus) => clouds.map((points, sequence) => {
      const id = `${corpus ? 'G' : 'R'}${sequence + 1}`;
      return `<g class="slice-sequence corpus-${corpus}" role="button" tabindex="0" aria-label="Follow ${id}, 12 token states" data-sequence="${corpus}-${sequence}" data-sequence-id="${id}"><ellipse class="cloud-outline"/><line class="slice-row"/>${points.map(() => '<line class="slice-guide"/><circle class="slice-ghost" r="2.6"/>').join('')}${points.map(() => '<circle class="slice-token" r="2.6"/>').join('')}<text class="cloud-label" text-anchor="middle">${id}</text></g>`;
    }).join('')).join('')}</svg><p class="embedding-key"><span class="corpus-0">● Reference</span><span class="corpus-1">● Generated</span><span>Same ξ · separate sequences</span></p>`;
    state = { element, corpora, direction: slices.direction, phase: 'clouds', elapsed: 0, frame: null, events: new AbortController() };
    state.svg = element.querySelector('.slicing-plot');
    state.axes = element.querySelector('.slice-axes'); state.axis = element.querySelector('.slice-direction'); state.rowAxes = element.querySelector('.slice-row-axes');
    state.toggle = element.querySelector('[data-slice-toggle]'); state.reset = element.querySelector('[data-slice-reset]'); state.status = element.querySelector('.slicing-status');
    state.rows = [corpora.reference, corpora.generated].flatMap((clouds, corpus) => clouds.map((points, sequence) => {
      const group = element.querySelector(`[data-sequence="${corpus}-${sequence}"]`);
      return { corpus, sequence, points, group, label: group.querySelector('text'), outline: group.querySelector('ellipse'), axis: group.querySelector('.slice-row'), dots: [...group.querySelectorAll('.slice-token')], ghosts: [...group.querySelectorAll('.slice-ghost')], guides: [...group.querySelectorAll('.slice-guide')] };
    }));
    states.set(element, state);
    const options = { signal: state.events.signal };
    state.toggle.addEventListener('click', () => state.phase === 'slicing' ? pause(state) : play(state), options);
    state.reset.addEventListener('click', () => {
      stop(state); state.elapsed = 0; state.phase = 'clouds'; paint(state); updateControls(state); state.toggle.focus();
    }, options);
    state.svg.addEventListener('keydown', event => {
      if (['Enter', ' '].includes(event.key) && event.target.matches('[data-sequence]')) {
        event.preventDefault(); event.target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      }
    }, options);
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(state); }, options);
    reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) finish(state); }, options);
    state.resize = new ResizeObserver(() => measure(state)); state.resize.observe(state.svg);
    measure(state);
    if (reducedMotion.matches) finish(state);
    updateControls(state);
  }
  state.rows.forEach(row => {
    row.active = row.corpus === traced.corpus && row.sequence === traced.sequence;
    row.group.classList.toggle('is-traced', row.active);
    row.group.setAttribute('aria-pressed', String(row.active));
    row.dots.forEach(dot => dot.setAttribute('r', row.active ? '3.3' : '2.6'));
  });
  paint(state);
}

export function playTokenSlicing(element) { const state = states.get(element); if (state) play(state); }
export const SLICING_DURATION = duration;
// Scroll-driven pages set the slicing time directly instead of playing it.
export function seekTokenSlicing(element, elapsed) {
  const state = states.get(element);
  if (!state) return;
  stop(state);
  state.elapsed = Math.max(0, Math.min(duration, elapsed));
  state.phase = state.elapsed <= 0 ? 'clouds' : state.elapsed >= duration ? 'rows' : 'paused';
  paint(state); updateControls(state);
}
export function pauseTokenSlicing(element) { const state = states.get(element); if (state) pause(state); }
export function disposeTokenSlicing(element) {
  const state = states.get(element);
  if (!state) return;
  stop(state); state.resize.disconnect(); state.events.abort(); states.delete(element);
  delete element.dataset.slicingPhase;
}
