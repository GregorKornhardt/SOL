// Illustrative text-to-cloud animation. Coordinates come from the toy corpus,
// not from encoding these example sentences in the browser.
import { cloudFrame } from './cloud-geometry.js';
const examples = [
  [
    'Small birds fly across the quiet lake as the sun rises.',
    'The team scored two late goals and won the final match.',
    'A new chip uses less power while running the same code.',
    'Warm bread and fresh fruit filled the market stalls this morning.',
    'The train left at dawn and reached the coast by noon.',
    'She read the last page and placed the book back down.',
  ],
  [
    'Bright stars shone above the still lake through the cold night.',
    'Fans cheered as their team took the lead before half time.',
    'The new phone loads apps fast and lasts all day long.',
    'Fresh soup and warm rolls were served beside the town square.',
    'A small boat crossed the bay and reached port after sunset.',
    'He closed the old book and set it near the window.',
  ],
];
const states = new WeakMap();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => x * x * (3 - 2 * x);
const duration = 3600;
const mix = (a, b, t) => a + (b - a) * t;

export function sourceText(corpus, sequence, scenario) {
  if (corpus && scenario === 'match') return examples[0][sequence];
  if (corpus && scenario === 'collapse') return examples[0][0];
  return examples[corpus][sequence];
}

function updateControls(state) {
  const { element, phase } = state;
  element.dataset.embeddingPhase = phase;
  element.querySelector('[data-map-toggle]').textContent = {
    texts: '▶ Map sequences into ℝ²', mapping: 'Ⅱ Pause mapping', paused: '▶ Continue mapping', clouds: '↺ Replay mapping',
  }[phase];
  element.querySelector('[data-map-texts]').hidden = phase === 'texts';
  element.querySelector('.embedding-status').textContent = {
    texts: '12 text sequences → 12 token clouds', mapping: 'Tokens move; sequence labels stay with their clouds.',
    paused: 'Mapping paused.', clouds: '12 clouds in one shared ℝ² space · one dot per token',
  }[phase];
  // Only the currently visible representation participates in keyboard navigation.
  state.rows.forEach(row => {
    row.source.inert = phase !== 'texts';
    row.cloud.setAttribute('tabindex', phase === 'clouds' ? '0' : '-1');
    row.cloud.setAttribute('aria-hidden', String(phase !== 'clouds'));
  });
}

function measure(state) {
  if (states.get(state.element) !== state) return;
  // Work in layout pixels: a page may scale the whole diagram with a CSS transform.
  const bounds = state.stage.getBoundingClientRect();
  const w = state.stage.offsetWidth, h = state.stage.offsetHeight;
  const scale = bounds.width / w || 1;
  const frame = cloudFrame(w, h);
  state.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  state.svg.querySelector('.embedding-axes').innerHTML = `<path d="M36 30V${h - 30}H${w - 14}"/><text x="${w - 19}" y="${h - 8}">x</text><text x="14" y="28">y</text><text x="${w - 16}" y="25" text-anchor="end">Token states in ℝ²</text>`;
  state.rows.forEach(row => {
    row.starts = [...row.source.querySelectorAll('.embedding-token')].map(token => {
      const r = token.getBoundingClientRect();
      return [(r.x - bounds.x + r.width / 2) / scale, (r.y - bounds.y + r.height / 2) / scale];
    });
    row.ends = row.points.map(frame.point);
    const xs = row.ends.map(p => p[0]), ys = row.ends.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    let rx = (Math.max(...xs) - Math.min(...xs)) / 2 + 8;
    let ry = (Math.max(...ys) - Math.min(...ys)) / 2 + 8;
    const fit = Math.max(1, ...row.ends.map(([x, y]) => Math.hypot((x - cx) / rx, (y - cy) / ry)));
    rx *= fit; ry *= fit;
    for (const [key, value] of Object.entries({ cx, cy, rx, ry })) row.outline.setAttribute(key, value);
    const r = row.source.querySelector('strong').getBoundingClientRect();
    row.labelStart = [(r.x - bounds.x + r.width / 2) / scale, (r.y - bounds.y + r.height * .8) / scale];
    row.labelEnd = [cx, cy - ry - 5];
  });
  paint(state, state.elapsed);
}

function paint(state, elapsed) {
  const finished = elapsed >= duration;
  state.svg.style.opacity = state.phase === 'texts' ? '0' : '1';
  state.svg.querySelector('.embedding-axes').style.opacity = clamp(elapsed / 600);
  state.rows.forEach(row => {
    const delay = (row.sequence * 2 + row.corpus) * 150;
    const progress = ease(clamp((elapsed - delay) / 1500));
    row.source.style.opacity = state.phase === 'texts' ? '1' : String(1 - progress);
    row.group.style.opacity = state.phase === 'texts' ? '0' : '1';
    row.circles.forEach((circle, i) => {
      const p = finished ? 1 : ease(clamp((elapsed - delay - i * 18) / 1400));
      const x = mix(row.starts[i][0], row.ends[i][0], p), y = mix(row.starts[i][1], row.ends[i][1], p);
      circle.setAttribute('cx', x); circle.setAttribute('cy', y);
      circle.style.opacity = clamp(p * 4);
      row.words[i].setAttribute('x', x); row.words[i].setAttribute('y', y + 4);
      row.words[i].style.opacity = 1 - clamp(p * 3);
    });
    row.label.setAttribute('x', mix(row.labelStart[0], row.labelEnd[0], progress));
    row.label.setAttribute('y', mix(row.labelStart[1], row.labelEnd[1], progress));
    row.outline.style.opacity = clamp((progress - .75) * 4);
  });
}

function stop(state) {
  cancelAnimationFrame(state.frame);
  state.frame = null;
}

function finish(state) {
  stop(state);
  state.elapsed = duration; state.phase = 'clouds';
  paint(state, duration); updateControls(state);
}

function pause(state) {
  if (state.phase !== 'mapping') return;
  state.elapsed = Math.min(duration, performance.now() - state.started);
  stop(state); state.phase = 'paused';
  paint(state, state.elapsed); updateControls(state);
}

function play(state) {
  stop(state);
  if (state.phase !== 'paused') state.elapsed = 0;
  state.phase = 'mapping';
  measure(state);
  if (reducedMotion.matches) { finish(state); return; }
  state.started = performance.now() - state.elapsed;
  updateControls(state);
  const frame = now => {
    state.elapsed = Math.min(duration, now - state.started);
    paint(state, state.elapsed);
    if (state.elapsed >= duration) finish(state);
    else state.frame = requestAnimationFrame(frame);
  };
  state.frame = requestAnimationFrame(frame);
}

export function disposeEmbedding(element) {
  const state = states.get(element);
  if (!state) return;
  stop(state); state.resize.disconnect(); state.events.abort();
  states.delete(element);
  delete element.dataset.embeddingPhase;
}

export function renderEmbedding(element, corpora, traced, scenario) {
  let state = states.get(element);
  if (state && state.corpora !== corpora) { disposeEmbedding(element); state = null; }
  if (!state) {
    const groups = [corpora.reference, corpora.generated];
    element.innerHTML = `<div class="embedding-controls"><button class="button button-small" data-map-toggle>▶ Map sequences into ℝ²</button><button class="text-button" data-map-texts hidden>Show texts</button></div><p class="embedding-status" role="status"></p><div class="embedding-stage"><div class="embedding-sources">${groups.map((clouds, corpus) => `<section class="corpus-group corpus-${corpus}"><h4>${corpus ? 'Generated' : 'Reference'} corpus<span>6 text sequences</span></h4><div class="sequence-cards">${clouds.map((points, sequence) => {
      const id = `${corpus ? 'G' : 'R'}${sequence + 1}`;
      const tokens = sourceText(corpus, sequence, scenario).match(/\w+|[.]/g);
      return `<button class="sequence-card embedding-sequence" data-sequence="${corpus}-${sequence}" data-sequence-id="${id}" aria-label="Follow ${id}: ${sourceText(corpus, sequence, scenario)}"><strong>${id}</strong><span class="embedding-sentence">${tokens.map((word, i) => `<span class="embedding-token">${word}</span>${i < tokens.length - 2 ? ' ' : ''}`).join('')}</span></button>`;
    }).join('')}</div></section>`).join('')}</div><svg class="embedding-plot" xmlns="http://www.w3.org/2000/svg" aria-label="A shared two-dimensional space containing one token cloud per sequence"><g class="embedding-axes"></g>${groups.map((clouds, corpus) => clouds.map((points, sequence) => {
      const id = `${corpus ? 'G' : 'R'}${sequence + 1}`;
      const tokens = sourceText(corpus, sequence, scenario).match(/\w+|[.]/g);
      return `<g class="embedding-moving corpus-${corpus}" data-cloud="${corpus}-${sequence}"><g class="embedding-cloud" role="button" tabindex="-1" aria-label="Follow ${id}, cloud of 12 token states" data-sequence="${corpus}-${sequence}" data-sequence-id="${id}"><ellipse class="cloud-outline"/>${points.map(() => '<circle class="embedding-dot" r="2.6"/>').join('')}<text class="cloud-label" text-anchor="middle">${id}</text></g>${tokens.map(word => `<text class="flying-token" text-anchor="middle">${word}</text>`).join('')}</g>`;
    }).join('')).join('')}</svg></div><p class="embedding-key"><span class="corpus-0">● Reference</span><span class="corpus-1">● Generated</span><span>One labeled cloud = one sequence</span></p>`;
    state = { element, corpora, phase: 'texts', elapsed: 0, frame: null, rows: [], events: new AbortController() };
    state.stage = element.querySelector('.embedding-stage');
    state.svg = element.querySelector('.embedding-plot');
    groups.forEach((clouds, corpus) => clouds.forEach((points, sequence) => {
      const group = element.querySelector(`[data-cloud="${corpus}-${sequence}"]`);
      state.rows.push({ corpus, sequence, points, group,
        source: element.querySelector(`.embedding-sequence[data-sequence="${corpus}-${sequence}"]`),
        cloud: group.querySelector('.embedding-cloud'), outline: group.querySelector('ellipse'),
        label: group.querySelector('.cloud-label'), circles: [...group.querySelectorAll('circle')], words: [...group.querySelectorAll('.flying-token')],
      });
    }));
    states.set(element, state);
    const options = { signal: state.events.signal };
    element.querySelector('[data-map-toggle]').addEventListener('click', () => state.phase === 'mapping' ? pause(state) : play(state), options);
    element.querySelector('[data-map-texts]').addEventListener('click', () => {
      stop(state); state.elapsed = 0; state.phase = 'texts'; paint(state, 0); updateControls(state);
      element.querySelector('[data-map-toggle]').focus();
    }, options);
    state.svg.addEventListener('keydown', event => {
      if (['Enter', ' '].includes(event.key) && event.target.matches('[data-sequence]')) {
        event.preventDefault(); event.target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      }
    }, options);
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(state); }, options);
    reducedMotion.addEventListener('change', () => { if (reducedMotion.matches && state.phase !== 'texts') finish(state); }, options);
    state.resize = new ResizeObserver(() => measure(state));
    state.resize.observe(state.stage);
    measure(state); updateControls(state);
  }
  state.rows.forEach(row => {
    const active = row.corpus === traced.corpus && row.sequence === traced.sequence;
    row.source.classList.toggle('is-traced', active);
    row.source.setAttribute('aria-pressed', String(active));
    row.cloud.classList.toggle('is-traced', active);
    row.cloud.setAttribute('aria-pressed', String(active));
    row.circles.forEach(circle => circle.setAttribute('r', active ? '3.3' : '2.6'));
  });
}

export function playEmbedding(element) {
  const state = states.get(element);
  if (state) play(state);
}

export function pauseEmbedding(element) {
  const state = states.get(element);
  if (state) pause(state);
}

export const EMBEDDING_DURATION = duration;
// Scroll-driven pages set the mapping time directly instead of playing it.
export function seekEmbedding(element, elapsed) {
  const state = states.get(element);
  if (!state) return;
  stop(state);
  state.elapsed = Math.max(0, Math.min(duration, elapsed));
  state.phase = state.elapsed <= 0 ? 'texts' : state.elapsed >= duration ? 'clouds' : 'paused';
  paint(state, state.elapsed); updateControls(state);
}
