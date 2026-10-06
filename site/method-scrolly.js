// Scroll-driven method walkthrough: the reading position sets each animation's time.
import { makeCorpora, sliceCorpora, toySOL } from './sol-demo.js';
import { renderMethodDiagram, seekQuantiles, seekTransport, QUANTILE_DURATION, TRANSPORT_DURATION } from './method-diagram.js';
import { seekEmbedding, EMBEDDING_DURATION } from './sequence-embedding.js';
import { seekTokenSlicing, SLICING_DURATION } from './token-slicing.js';
import { formulas, stages } from './method-content.js';
import { $, $$ } from './dom.js';

// Each step scrubs one stage's animation between two times in milliseconds.
// 650 and 2800 are the token-slicing phase boundaries; 3300 starts the integration
// against g; 3000 is where the transport picture's score starts counting.
const steps = [
  { stage: 0, from: 0, to: 0 },
  { stage: 0, from: 0, to: EMBEDDING_DURATION },
  { stage: 1, from: 0, to: 650 },
  { stage: 1, from: 650, to: 2800 },
  { stage: 1, from: 2800, to: SLICING_DURATION },
  { stage: 2, from: 0, to: 3300 },
  { stage: 2, from: 3300, to: QUANTILE_DURATION },
  { stage: 3, from: 0, to: 3000 },
  { stage: 3, from: 3000, to: TRANSPORT_DURATION },
];
const seekers = [seekEmbedding, seekTokenSlicing, seekQuantiles, seekTransport];
const colors = ['#c1f65c', '#ec91c5'];
const explanations = {
  shift: 'The token clouds of the generated sequences move away from those of the reference sequences. SOL measures the resulting corpus difference.',
  match: 'The two synthetic corpora contain identical sequence clouds, so their SOL distance is zero.',
  collapse: 'All six generated sequences copy the first reference cloud. SOL responds to the sequence variety that is now missing.',
  spread: 'Each generated sequence’s token cloud contracts around its center. SOL responds to the changed token distributions.',
};
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const clamp = x => Math.max(0, Math.min(1, x));

let corpora = makeCorpora(.55);
let score = toySOL(corpora.reference, corpora.generated);
let traced = { corpus: 0, sequence: 0 };
let renderedStage = -1;
let pending = false;

function render(stage) {
  const diagram = $('#method-diagram');
  const slices = sliceCorpora(corpora.reference, corpora.generated);
  const result = renderMethodDiagram(diagram, corpora, slices, stage, traced, $('#scenario').value, score);
  renderedStage = stage;
  $('#method-visual-caption').textContent = stages[stage].caption;
  $('#scrolly-trace').textContent = `Following ${result.id}: “${result.sentence}”`;
  $('#scrolly-trace').style.setProperty('--trace-color', colors[traced.corpus]);
  $('#trace-select').value = `${traced.corpus}-${traced.sequence}`;
  fit();
}

// Scale the diagram down to fit the pinned panel when it is still too tall.
// A negative bottom margin removes the space a scaled diagram no longer
// covers, so the caption follows it directly.
function fit() {
  const diagram = $('#method-diagram'), stage = $('.scrolly-stage'), panel = $('.scrolly-panel');
  const width = stage.clientWidth;
  // Wide screens cap the panel at the window below the header; phones fix its height.
  const limit = panel.offsetWidth < innerWidth * .7 ? innerHeight - parseFloat(getComputedStyle(panel).top) : panel.clientHeight;
  const style = getComputedStyle(stage);
  const above = stage.getBoundingClientRect().top - panel.getBoundingClientRect().top;
  const available = limit - above - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) - $('#method-visual-caption').offsetHeight - 2;
  const layoutWidth = width;
  diagram.style.transform = ''; diagram.style.marginBottom = '';
  diagram.style.width = `${layoutWidth}px`;
  const height = diagram.offsetHeight;
  const scale = Math.min(1, width / layoutWidth, available / height);
  diagram.style.transform = scale < 1 ? `translateX(${(width - layoutWidth * scale) / 2}px) scale(${scale})` : '';
  diagram.style.marginBottom = scale < 1 ? `${-(1 - scale) * height}px` : '';
}

// The reading focus sits just above the middle of the text area: of the window
// beside the panel on wide screens, of the band below the pinned panel on narrow ones.
function focusPoint() {
  const panel = $('.scrolly-panel');
  if (panel.offsetWidth < innerWidth * .7) return innerHeight * .45;
  const bottom = parseFloat(getComputedStyle(panel).top) + panel.offsetHeight;
  return bottom + (innerHeight - bottom) * .5;
}

// Vertical center of a step's visible text, not of its taller block.
export function textBounds(step) {
  const boxes = [...step.children].map(child => child.getBoundingClientRect()).filter(box => box.height > 0);
  return { top: boxes[0].top, bottom: boxes.at(-1).bottom };
}
function textCenter(step) {
  const { top, bottom } = textBounds(step);
  return (top + bottom) / 2;
}

// Each step owns the focus from halfway to the previous text to halfway to the
// next one. Its animation finishes when its text reaches the focus and then
// holds until the next text is closer.
export function scrollyPosition() {
  const focus = focusPoint();
  const centers = $$('.scrolly-step').map(textCenter);
  const gap = i => centers[Math.min(i + 1, centers.length - 1)] - centers[Math.max(i, 1) - 1];
  const start = i => (i ? (centers[i - 1] + centers[i]) / 2 : centers[0] - gap(0) / 2);
  const end = i => (i < centers.length - 1 ? (centers[i] + centers[i + 1]) / 2 : centers[i] + gap(i) / 2);
  let index = 0;
  centers.forEach((_, i) => { if (start(i) <= focus) index = i; });
  return { index, progress: clamp((focus - start(index)) / (end(index) - start(index))), focus, center: centers[index] };
}

function update() {
  pending = false;
  const { index, progress } = scrollyPosition();
  const amount = reducedMotion.matches ? 1 : clamp((progress - .05) / .45);
  const step = steps[index];
  if (step.stage !== renderedStage) render(step.stage);
  seekers[step.stage]($('#method-diagram'), step.from + amount * (step.to - step.from));
  $('#scrolly').dataset.step = String(index);
  $$('.scrolly-step').forEach((element, i) => element.classList.toggle('is-active', i === index));
  $$('.scrolly-progress a').forEach((link, i) => {
    if (i === index) link.setAttribute('aria-current', 'step');
    else link.removeAttribute('aria-current');
  });
}

function schedule() {
  if (!pending) { pending = true; requestAnimationFrame(update); }
}

function setTrace(value) {
  const [corpus, sequence] = value.split('-').map(Number);
  traced = { corpus, sequence };
  render(renderedStage);
  update();
}

function setScenario() {
  corpora = makeCorpora(.55, $('#scenario').value);
  score = toySOL(corpora.reference, corpora.generated);
  $('#scenario-explanation').textContent = explanations[$('#scenario').value];
  $('#method-sol').textContent = score.toFixed(4);
  renderedStage = -1;
  update();
}

export function initMethodScrolly() {
  $$('.scrolly [data-formula]').forEach(element => { element.innerHTML = formulas[Number(element.dataset.formula)]; });
  const header = $('.site-header');
  const setHeader = () => document.documentElement.style.setProperty('--header-height', `${header.offsetHeight}px`);
  setHeader();
  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', () => { setHeader(); schedule(); });
  new ResizeObserver(() => { fit(); schedule(); }).observe($('.scrolly-stage'));
  reducedMotion.addEventListener('change', schedule);
  $('#trace-select').addEventListener('change', event => setTrace(event.target.value));
  $('#scenario').addEventListener('change', setScenario);
  $('#method-diagram').addEventListener('click', event => {
    const button = event.target.closest('[data-sequence]');
    if (button) setTrace(button.dataset.sequence);
  });
  // Progress links bring their step's text to the focus, where its animation is complete.
  $$('.scrolly-progress a').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    const target = $(link.getAttribute('href'));
    const top = scrollY + textCenter(target) - focusPoint();
    scrollTo({ top, behavior: reducedMotion.matches ? 'auto' : 'smooth' });
    history.replaceState(null, '', link.getAttribute('href'));
  }));
  setScenario();
  document.fonts.ready.then(() => { renderedStage = -1; update(); });
}
