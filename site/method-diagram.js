// Animated text embedding and token slicing, followed by the shared scalar plots.
// Steps 3 and 4 show their standard pictures unless an animation time is set.
import { renderEmbedding, disposeEmbedding, sourceText } from './sequence-embedding.js';
import { renderTokenSlicing, disposeTokenSlicing } from './token-slicing.js';
const colors = ['#c1f65c', '#ec91c5'];
const plots = new WeakMap();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const quantileDuration = 6600;
const transportDuration = 3800;
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { const t = clamp(x); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;

// Step-3 animation: token rows -> sorted dots -> curves -> sweep against g -> scalars.
function quantileFrame(elapsed) {
  return {
    rows: 1 - ease((elapsed - 600) / 700),
    move: rank => ease((elapsed - 600 - rank * 40) / 1300),
    axes: ease((elapsed - 600) / 900),
    curves: ease((elapsed - 2300) / 1000),
    dots: 1 - ease((elapsed - 3300) / 500),
    gp: ease((elapsed - 3300) / 500),
    sweep: clamp((elapsed - 3800) / 1600),
    scalars: ease((elapsed - 5400) / 500),
    drop: order => ease((elapsed - 5500 - order * 40) / 600),
  };
}

// Step-4 animation: rank labels in sorted order -> rank-matching lines -> averaged score.
function transportFrame(elapsed) {
  return {
    rank: i => ease((elapsed - 300 - i * 170) / 350),
    link: i => ease((elapsed - 1500 - i * 170) / 500),
    focus: ease((elapsed - 2600) / 400),
    score: ease((elapsed - 3000) / 800),
  };
}

function line(ctx, x1, y1, x2, y2, color, width = 1) {
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}
function dot(ctx, x, y, color, size = 3) {
  ctx.fillStyle = color; ctx.fillRect(x - size / 2, y - size / 2, size, size);
}
function label(ctx, text, x, y, color = '#92a67b', size = 10, align = 'left') {
  ctx.font = `${Math.max(10, size)}px Consolas, monospace`;
  ctx.fillStyle = color; ctx.textAlign = align; ctx.fillText(text, x, y);
}

function sequenceTrace(slices, traced) {
  const groups = [slices.reference, slices.generated];
  const rank = groups[traced.corpus].ranked.findIndex(row => row.sequence === traced.sequence);
  const partner = groups[1 - traced.corpus].ranked[rank];
  const value = groups[traced.corpus].sequenceValues[traced.sequence];
  return {
    id: `${traced.corpus ? 'G' : 'R'}${traced.sequence + 1}`,
    rank, value,
    partnerId: `${traced.corpus ? 'R' : 'G'}${partner.sequence + 1}`,
    partnerValue: partner.value,
    gap: Math.abs(value - partner.value),
  };
}

function highlightDot(ctx, x, y, color) {
  dot(ctx, x, y, color, 7);
  ctx.beginPath(); ctx.arc(x, y, 9, 0, 2 * Math.PI);
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
}

function drawClassic(canvas, slices, stage, score, traced, elapsed = null) {
  // Layout pixels, so a CSS transform on the page scales the drawing as a whole.
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h || !canvas.isConnected) return;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const trace = sequenceTrace(slices, traced);
  const groups = [slices.reference, slices.generated];
  const selectedColor = colors[traced.corpus];
  if (stage === 2) {
    // Without an animation time, draw the standard picture.
    const frame = elapsed === null ? null : quantileFrame(elapsed);
    const fade = weight => { ctx.globalAlpha = frame ? weight : 1; };
    // Narrow and tall: g(u) below the curves. Short panels shrink the curve plot.
    const compact = w < 420 && h >= 400;
    const plotRight = compact ? w - 24 : w * .60;
    const plotBottom = compact ? 150 : h >= 390 ? 185 : Math.round(h * .42);
    const quantiles = groups.flatMap(s => s.quantiles.flat());
    const low = Math.min(...quantiles), high = Math.max(...quantiles);
    const qx = m => 34 + m / (slices.gp.length - 1) * (plotRight - 34);
    const qy = value => plotBottom - 12 - (value - low) / (high - low || 1) * (plotBottom - 50);
    fade(frame?.axes);
    line(ctx, 33, plotBottom, plotRight, plotBottom, '#5c7144');
    line(ctx, 33, plotBottom, 33, 24, '#5c7144');
    label(ctx, 'Q(u)', 12, 15, '#becbb1', 12);
    label(ctx, '0', 32, plotBottom + 17, '#becbb1', 11);
    label(ctx, '1', plotRight, plotBottom + 17, '#becbb1', 11, 'right');
    // Step 2's rows: projected token values on the shared scale from -2 to 2.
    const rowX = value => 48 + (value + 2) / 4 * (w - 68);
    const rowGap = Math.min(25, (h - 100) / 13);
    const rowY = (c, i) => 30 + i * rowGap + c * (6 * rowGap + 30);
    if (frame) {
      ctx.globalAlpha = frame.rows;
      groups.forEach((s, c) => s.tokens.forEach((_, i) => {
        const active = c === traced.corpus && i === traced.sequence;
        line(ctx, 42, rowY(c, i), w - 18, rowY(c, i), `${colors[c]}${active ? '66' : '33'}`, 12);
        label(ctx, `${c ? 'G' : 'R'}${i + 1}`, 22, rowY(c, i) + 4, colors[c], 12, 'center');
      }));
      [-2, -1, 0, 1, 2].forEach(value => label(ctx, String(value), rowX(value), rowY(1, 5) + 34, '#becbb1', 11, 'center'));
    }
    const drawCurve = (q, color, active) => {
      ctx.strokeStyle = active ? color : `${color}55`; ctx.lineWidth = active ? 2.8 : 1.2; ctx.beginPath();
      q.forEach((value, m) => {
        if (!m) ctx.moveTo(qx(m), qy(value)); else ctx.lineTo(qx(m), qy(value));
      }); ctx.stroke();
    };
    const selectedQuantiles = groups[traced.corpus].quantiles[traced.sequence];
    const reveal = frame ? frame.curves : 1;
    if (reveal > 0) {
      ctx.globalAlpha = 1;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, 34 + reveal * (plotRight - 34) + 2, h); ctx.clip();
      groups.forEach((s, c) => s.quantiles.forEach((q, i) => {
        if (c !== traced.corpus || i !== traced.sequence) drawCurve(q, colors[c], false);
      }));
      drawCurve(selectedQuantiles, selectedColor, true);
      ctx.restore();
    }
    fade(reveal);
    label(ctx, trace.id, plotRight, qy(selectedQuantiles.at(-1)) - 10, selectedColor, 13, 'right');
    if (frame) {
      // Rank sets each token's fraction u; its value becomes the height Q(u).
      ctx.globalAlpha = frame.dots;
      const tokenDots = (c, i, active) => groups[c].tokens[i].forEach((value, rank, values) => {
        const t = frame.move(rank);
        const bin = (rank + .5) * slices.gp.length / values.length - .5;
        dot(ctx, mix(rowX(value), qx(bin), t), mix(rowY(c, i), qy(value), t), active ? colors[c] : `${colors[c]}88`, active ? 5 : 3);
      });
      groups.forEach((s, c) => s.tokens.forEach((_, i) => {
        if (c !== traced.corpus || i !== traced.sequence) tokenDots(c, i, false);
      }));
      tokenDots(traced.corpus, traced.sequence, true);
    }
    fade(frame?.gp);
    const M = slices.gp.length;
    const gpLeft = compact ? 34 : w * .72, gpRight = w - 24;
    const gpMiddle = compact ? 227 : 94;
    label(ctx, 'Shared g(u)', (gpLeft + gpRight) / 2, gpMiddle - 38, '#d3ddc6', 12, 'center');
    const gpMax = Math.max(...slices.gp.map(Math.abs));
    const gx = u => gpLeft + u * (gpRight - gpLeft), gy = v => gpMiddle - v / gpMax * 26;
    line(ctx, gpLeft, gpMiddle, gpRight, gpMiddle, '#5c714477');
    ctx.beginPath(); ctx.strokeStyle = '#dbe5cd'; ctx.lineWidth = 1.4;
    slices.gp.forEach((v, i) => {
      if (!i) ctx.moveTo(gx(i / (M - 1)), gy(v)); else ctx.lineTo(gx(i / (M - 1)), gy(v));
    }); ctx.stroke();
    // Sweep u over both plots while the integral of Q·g accumulates.
    const sweep = frame ? frame.sweep : 1;
    if (frame && sweep > 0 && sweep < 1) {
      const m = Math.min(M - 1, Math.floor(sweep * M));
      const x = qx(sweep * (M - 1)), xg = gx(sweep);
      ctx.globalAlpha = 1;
      line(ctx, x, plotBottom, x, 24, '#dbe5cd66');
      dot(ctx, x, qy(selectedQuantiles[m]), selectedColor, 7);
      line(ctx, xg, gpMiddle - 30, xg, gpMiddle + 30, '#dbe5cd66');
      dot(ctx, xg, gy(slices.gp[m]), '#dbe5cd', 6);
      fade(frame.gp);
    }
    const partial = selectedQuantiles.reduce((sum, q, m) => sum + q * slices.gp[m] * clamp(sweep * M - m), 0) / M;
    label(ctx, sweep < 1 ? `${trace.id}: ∫₀ᵘ Q(u) g(u) du = ${partial.toFixed(3)}` : `${trace.id}: ∫ Q(u) g(u) du = ${trace.value.toFixed(3)}`,
      w / 2, compact ? 292 : plotBottom + 47, selectedColor, 12, 'center');
    const values = groups.flatMap(s => s.sequenceValues);
    const lo = Math.min(...values) - .12, hi = Math.max(...values) + .12;
    const map = v => 35 + (v - lo) / (hi - lo) * (w - 60);
    const scalarY = c => h - 96 + c * 42;
    fade(frame?.scalars);
    groups.forEach((s, c) => {
      line(ctx, 28, scalarY(c), w - 22, scalarY(c), `${colors[c]}55`);
      label(ctx, c ? 'G' : 'R', 8, scalarY(c) + 4, colors[c], 12);
    });
    label(ctx, 'Scalar value · one dot per sequence', w / 2, h - 13, '#becbb1', 11, 'center');
    // Each curve's end drops onto its corpus row as that sequence's scalar.
    ctx.globalAlpha = 1;
    const drop = (c, i) => (frame ? frame.drop(c * groups[0].sequenceValues.length + i) : 1);
    const scalarPoint = (c, i, t) => [mix(qx(M - 1), map(groups[c].sequenceValues[i]), t), mix(qy(groups[c].quantiles[i].at(-1)), scalarY(c), t)];
    groups.forEach((s, c) => s.sequenceValues.forEach((_, i) => {
      if ((c !== traced.corpus || i !== traced.sequence) && drop(c, i) > 0) dot(ctx, ...scalarPoint(c, i, drop(c, i)), `${colors[c]}88`, 4);
    }));
    const tracedDrop = drop(traced.corpus, traced.sequence);
    if (tracedDrop >= 1) {
      highlightDot(ctx, map(trace.value), scalarY(traced.corpus), selectedColor);
      label(ctx, trace.id, map(trace.value), scalarY(traced.corpus) - 16, selectedColor, 12, 'center');
    } else if (tracedDrop > 0) dot(ctx, ...scalarPoint(traced.corpus, traced.sequence, tracedDrop), selectedColor, 6);
    ctx.globalAlpha = 1;
  } else {
    const frame = elapsed === null ? null : transportFrame(elapsed);
    const weight = value => { ctx.globalAlpha = frame ? value : 1; };
    const values = [...slices.reference.values, ...slices.generated.values];
    const lo = Math.min(...values) - .15, hi = Math.max(...values) + .15;
    const map = v => 35 + (v - lo) / (hi - lo) * (w - 70);
    weight(frame?.focus);
    label(ctx, `${trace.id} ↔ ${trace.partnerId} · rank ${trace.rank + 1}`, w / 2, 23, '#dbe5cd', 14, 'center');
    ctx.globalAlpha = 1;
    // Short panels drop the subtitle and move the rows apart from the labels.
    const short = h < 360;
    if (!short) label(ctx, 'Match by rank · keep sequence IDs', w / 2, 45, '#becbb1', w < 420 ? 11 : 12, 'center');
    const rowY = c => h * (c ? (short ? .6 : .65) : (short ? .33 : .30));
    // Rank-matching lines draw one rank at a time: the exact 1D transport plan.
    const link = (i, color, width) => {
      const t = frame ? frame.link(i) : 1;
      if (t <= 0) return;
      const x1 = map(slices.reference.values[i]), x2 = map(slices.generated.values[i]);
      line(ctx, x1, rowY(0), mix(x1, x2, t), mix(rowY(0), rowY(1), t), color, width);
    };
    slices.reference.values.forEach((v, i) => { if (i !== trace.rank) link(i, '#afc88944', 1.2); });
    link(trace.rank, '#dbe5cd', 2.7);
    groups.forEach((s, c) => {
      const y = rowY(c);
      line(ctx, 28, y, w - 28, y, `${colors[c]}44`);
      s.values.forEach((v, i) => {
        dot(ctx, map(v), y, i === trace.rank ? colors[c] : `${colors[c]}88`, 5);
        weight(frame?.rank(i));
        label(ctx, String(i + 1), map(v), y + (c ? 19 + i % 2 * 13 : -15 - i % 2 * 13), i === trace.rank ? colors[c] : '#94a485', 11, 'center');
        ctx.globalAlpha = 1;
      });
      const value = s.ranked[trace.rank];
      weight(frame?.focus);
      highlightDot(ctx, map(value.value), y, colors[c]);
      label(ctx, `${c ? 'G' : 'R'}${value.sequence + 1}`, map(value.value), y + (c ? 49 : -43), colors[c], 13, 'center');
      ctx.globalAlpha = 1;
    });
    // The score averages all slice pairs; it counts up as it appears.
    weight(frame?.score);
    label(ctx, 'TOY SOL · ALL 768 SLICE PAIRS', w / 2, h - 57, '#becbb1', 11, 'center');
    ctx.font = '25px SolPixel, monospace'; ctx.fillStyle = colors[0]; ctx.textAlign = 'center';
    ctx.fillText((frame ? score * frame.score : score).toFixed(3), w / 2, h - 24);
    ctx.globalAlpha = 1;
  }
}


function redraw(plot) {
  const animating = ['playing', 'paused'].includes(plot.anim.phase);
  const elapsed = plot.stage === 2 ? (animating ? plot.anim.elapsed : null) : plot.transport ?? null;
  drawClassic(plot.canvas, plot.slices, plot.stage, plot.score, plot.traced, elapsed);
}

function updateQuantileControls(plot) {
  const { anim } = plot;
  plot.element.dataset.quantilePhase = anim.phase;
  plot.toggle.textContent = {
    idle: '▶ Animate from token rows', playing: 'Ⅱ Pause animation', paused: '▶ Continue animation', done: '↺ Replay animation',
  }[anim.phase];
  plot.reset.hidden = !['playing', 'paused'].includes(anim.phase);
  const status = anim.phase === 'paused' ? 'Animation paused.'
    : anim.phase !== 'playing' ? '12 quantile curves · one shared g(u)'
      : anim.elapsed < 600 ? 'Start from step 2: one row of projected token values per sequence.'
        : anim.elapsed < 2300 ? 'Sort each row: rank sets the fraction u, value sets the height Q(u).'
          : anim.elapsed < 3300 ? 'Join the sorted values into one quantile curve per sequence.'
            : anim.elapsed < 3800 ? 'Bring in the shared Gaussian-process draw g(u).'
              : anim.elapsed < 5400 ? 'Sweep u from 0 to 1 while the integral of Q(u) g(u) accumulates.'
                : 'Each curve becomes one scalar on its corpus row.';
  if (plot.status.textContent !== status) plot.status.textContent = status;
}

function stopQuantiles(plot) { cancelAnimationFrame(plot.anim.frame); plot.anim.frame = null; }
function settleQuantiles(plot, phase) {
  stopQuantiles(plot); Object.assign(plot.anim, { phase, elapsed: 0 });
  redraw(plot); updateQuantileControls(plot);
}
function pauseQuantileAnimation(plot) {
  if (plot.anim.phase !== 'playing') return;
  plot.anim.elapsed = Math.min(quantileDuration, performance.now() - plot.anim.started);
  stopQuantiles(plot); plot.anim.phase = 'paused'; redraw(plot); updateQuantileControls(plot);
}
function playQuantileAnimation(plot) {
  if (plot.stage !== 2) return;
  stopQuantiles(plot);
  if (plot.anim.phase !== 'paused') plot.anim.elapsed = 0;
  if (reducedMotion.matches) { settleQuantiles(plot, 'done'); return; }
  plot.anim.phase = 'playing';
  plot.anim.started = performance.now() - plot.anim.elapsed;
  redraw(plot); updateQuantileControls(plot);
  const tick = now => {
    plot.anim.elapsed = Math.min(quantileDuration, now - plot.anim.started);
    if (plot.anim.elapsed >= quantileDuration) { settleQuantiles(plot, 'done'); return; }
    redraw(plot); updateQuantileControls(plot);
    plot.anim.frame = requestAnimationFrame(tick);
  };
  plot.anim.frame = requestAnimationFrame(tick);
}

function disposeClassic(element) {
  const plot = plots.get(element);
  if (plot) { stopQuantiles(plot); plot.resize.disconnect(); plot.events.abort(); }
  plots.delete(element);
  delete element.dataset.quantilePhase;
  delete element.dataset.transportPhase;
}

export function renderMethodDiagram(element, corpora, slices, stage, traced, scenario, score) {
  element.dataset.stage = stage;
  if (stage === 0) {
    disposeClassic(element);
    disposeTokenSlicing(element);
    renderEmbedding(element, corpora, traced, scenario);
  } else if (stage === 1) {
    disposeClassic(element);
    disposeEmbedding(element);
    renderTokenSlicing(element, corpora, slices, traced);
  } else {
    disposeEmbedding(element);
    disposeTokenSlicing(element);
    let plot = plots.get(element);
    if (!plot) {
      element.innerHTML = '<div class="embedding-controls quantile-controls"><button class="button button-small" data-quantile-toggle>▶ Animate from token rows</button><button class="text-button" data-quantile-reset hidden>Show curves</button></div><p class="embedding-status quantile-status" role="status"></p><div class="classic-legend legend"><span><i></i>Reference</span><span><i class="pink-dot"></i>Generated</span></div><canvas id="method-canvas" role="img"></canvas>';
      plot = {
        element, canvas: element.querySelector('canvas'), controls: element.querySelector('.quantile-controls'),
        toggle: element.querySelector('[data-quantile-toggle]'), reset: element.querySelector('[data-quantile-reset]'),
        status: element.querySelector('.quantile-status'), anim: { phase: 'idle', elapsed: 0, frame: null }, events: new AbortController(),
      };
      const options = { signal: plot.events.signal };
      plot.toggle.addEventListener('click', () => plot.anim.phase === 'playing' ? pauseQuantileAnimation(plot) : playQuantileAnimation(plot), options);
      plot.reset.addEventListener('click', () => { settleQuantiles(plot, 'idle'); plot.toggle.focus(); }, options);
      document.addEventListener('visibilitychange', () => { if (document.hidden) pauseQuantileAnimation(plot); }, options);
      reducedMotion.addEventListener('change', () => {
        if (reducedMotion.matches && ['playing', 'paused'].includes(plot.anim.phase)) settleQuantiles(plot, 'done');
      }, options);
      plot.resize = new ResizeObserver(() => redraw(plot));
      plots.set(element, plot);
      plot.resize.observe(plot.canvas);
    }
    // Each visit to step 3 starts from the standard picture.
    if (plot.stage !== stage) {
      stopQuantiles(plot); Object.assign(plot.anim, { phase: 'idle', elapsed: 0 });
      plot.transport = null; delete element.dataset.transportPhase;
    }
    Object.assign(plot, { slices, stage, score, traced });
    plot.controls.hidden = plot.status.hidden = stage !== 2;
    if (stage === 2) updateQuantileControls(plot);
    else delete element.dataset.quantilePhase;
    const trace = sequenceTrace(slices, traced);
    const descriptions = {
      2: `Twelve quantile curves on one shared plot and a shared Gaussian-process function. Highlighted ${trace.id} becomes scalar ${trace.value.toFixed(3)}. Its labeled dot keeps that value in the ${traced.corpus ? 'generated' : 'reference'} corpus.`,
      3: `Two sorted scalar distributions, with six rank-matching lines between reference and generated sequences. Highlighted ${trace.id} matches ${trace.partnerId} at rank ${trace.rank + 1}. Toy SOL across all 768 slice pairs: ${score.toFixed(3)}.`,
    };
    plot.canvas.setAttribute('aria-label', descriptions[stage]);
    redraw(plot);
  }
  const id = `${traced.corpus ? 'G' : 'R'}${traced.sequence + 1}`;
  const trace = sequenceTrace(slices, traced);
  const descriptions = [
    `${id} is one of six sequences in the ${traced.corpus ? 'generated' : 'reference'} corpus. Its 12 tokens move into the highlighted cloud. Its label follows it through every step.`,
    `${id} keeps its 12 tokens through the projection. The guide lines show how those tokens reach ξ; the labeled row contains their projected values. Every sequence uses this same direction.`,
    `${id} is the highlighted quantile curve. Integrating it against the shared g(u) gives ${trace.value.toFixed(3)} for this pair of directions. The highlighted ${id} dot below is that same sequence, now represented by one scalar.`,
    `${id} has value ${trace.value.toFixed(3)} and moves to rank ${trace.rank + 1} after sorting. It is matched to ${trace.partnerId}, whose value is ${trace.partnerValue.toFixed(3)}, at the same rank. Their gap is ${trace.gap.toFixed(3)}. The match follows rank, while the labels retain the original sequence IDs.`,
  ];
  return {
    id,
    trace: descriptions[stage],
    sentence: sourceText(traced.corpus, traced.sequence, scenario),
  };
}

export function playQuantiles(element) { const plot = plots.get(element); if (plot) playQuantileAnimation(plot); }
export function pauseQuantiles(element) { const plot = plots.get(element); if (plot) pauseQuantileAnimation(plot); }
export const QUANTILE_DURATION = quantileDuration;
export const TRANSPORT_DURATION = transportDuration;
// Scroll-driven pages set the step-4 animation time; its end is the standard picture.
export function seekTransport(element, elapsed) {
  const plot = plots.get(element);
  if (!plot || plot.stage !== 3) return;
  plot.transport = elapsed >= transportDuration ? null : Math.max(0, elapsed);
  element.dataset.transportPhase = plot.transport === null ? 'done' : 'paused';
  redraw(plot);
}
// Scroll-driven pages set the animation time; its end is the standard picture.
export function seekQuantiles(element, elapsed) {
  const plot = plots.get(element);
  if (!plot || plot.stage !== 2) return;
  stopQuantiles(plot);
  Object.assign(plot.anim, elapsed >= quantileDuration ? { phase: 'idle', elapsed: 0 } : { phase: 'paused', elapsed: Math.max(0, elapsed) });
  redraw(plot); updateQuantileControls(plot);
}
