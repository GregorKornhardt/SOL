const $ = selector => document.querySelector(selector);
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const names = { dream: 'Dream 7B', gpt2: 'GPT-2 Large' };
const generators = { gpt2: 'GPT-2', 'gpt2-medium': 'GPT-2 Medium', 'gpt2-large': 'GPT-2 Large', 'gpt2-xl': 'GPT-2 XL' };
const colors = ['#c1f65c', '#ec91c5', '#80cfe4'];
const lineStyles = ['', '8 5', '2 5'];
let data;
let experiment = 'temperature';
let selected = 3;
let chart = null;
let dragPointer = null;
const number = v => v.toFixed(3);
const tableNumber = v => v.toFixed(4);
const percent = v => Number(v.toFixed(2)).toString();
const topicMixture = row => Object.entries(row.topic_shares).map(([topic, share]) => `${topic} ${percent(share)}%`).join(' · ');

// Ticks on a log axis: 1-2-5 per decade, or finer steps when the range is narrow.
function logTicks(low, high) {
  let ticks = [];
  for (const steps of [[1, 2, 5], [1, 1.5, 2, 3, 5, 7], [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]]) {
    ticks = [];
    for (let power = Math.floor(Math.log10(low)); power <= Math.ceil(Math.log10(high)); power++) {
      for (const step of steps) {
        const value = Number((step * 10 ** power).toPrecision(6));
        if (value >= low && value <= high) ticks.push(value);
      }
    }
    if (ticks.length >= 4) break;
  }
  if (ticks.length >= 3) return ticks;
  // Too narrow for log steps: evenly spaced round values.
  const rough = (high - low) / 4, magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].find(v => v * magnitude >= rough) * magnitude;
  return Array.from({ length: Math.floor(high / step) - Math.ceil(low / step) + 1 }, (_, i) => Number(((Math.ceil(low / step) + i) * step).toPrecision(6)));
}

function drawChart() {
  if (!chart) return;
  const { rows, series, label, xLabel } = chart;
  const width = Math.max(270, Math.round($('#evidence-chart').clientWidth));
  const height = 310, left = 50, right = 22, top = 35, bottom = 58;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const values = series.flatMap(s => rows.map(r => s.value(r)));
  let y, tickValues;
  if (chart.log) {
    const low = Math.min(...values) / 1.12, high = Math.max(...values) * 1.12;
    y = v => top + plotHeight * (1 - Math.log(v / low) / Math.log(high / low));
    tickValues = logTicks(low, high);
  } else {
    const highest = Math.max(...values) * 1.1;
    const magnitude = 10 ** Math.floor(Math.log10(highest / 4));
    const step = [1, 2, 2.5, 5, 10].find(v => v * magnitude >= highest / 4) * magnitude;
    const ticks = Math.ceil(highest / step), maximum = ticks * step;
    y = v => top + plotHeight * (1 - v / maximum);
    tickValues = Array.from({ length: ticks + 1 }, (_, tick) => step * tick);
  }
  const xValues = rows.map(chart.xValue);
  const x = i => left + (xValues[i] - xValues[0]) / (xValues.at(-1) - xValues[0]) * plotWidth;
  chart.positions = rows.map((_, i) => x(i));
  chart.bounds = { left, right: width - right, top, bottom: height - bottom };
  let content = `<title>${escape(label)}</title><desc>${escape($('#evidence-summary').textContent)} Exact values are available in the table below.</desc>`;
  for (const value of tickValues) {
    content += `<line x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}" stroke="#34422a"/><text x="${left-8}" y="${y(value)+4}" text-anchor="end">${number(value).replace(/0+$/, '').replace(/\.$/, '')}</text>`;
  }
  content += `<text x="${left}" y="17">${escape(label)}</text>`;
  rows.forEach((row, i) => {
    if (!chart.showTick || chart.showTick(row)) content += `<text x="${x(i)}" y="${height-34}" text-anchor="middle">${escape(xLabel(row))}</text>`;
  });
  content += `<text x="${left + plotWidth / 2}" y="${height-8}" text-anchor="middle">${escape(chart.axis)}</text>`;
  content += `<line x1="${x(selected)}" x2="${x(selected)}" y1="${top}" y2="${height-bottom}" stroke="#9ba68e" stroke-dasharray="3 5"/>`;
  series.forEach((s, index) => {
    const color = colors[index];
    const path = rows.map((r, i) => `${i ? 'L' : 'M'} ${x(i)} ${y(s.value(r))}`).join(' ');
    content += `<path data-series="${s.id || 'sol'}" d="${path}" fill="none" stroke="${color}" stroke-width="2.5" ${lineStyles[index] ? `stroke-dasharray="${lineStyles[index]}"` : ''}/>`;
    rows.forEach((r, i) => { content += `<circle data-evidence-index="${i}" data-series="${s.id || 'sol'}" cx="${x(i)}" cy="${y(s.value(r))}" r="${i === selected ? 6 : 4}" fill="${color}" stroke="#10120f" stroke-width="2"/>`; });
  });
  $('#evidence-chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${content}</svg>`;
  $('#evidence-chart').setAttribute('aria-label', `${chart.axis}. ${label}`);
}

function render() {
  if (!data) return;
  const isTemp = experiment === 'temperature', isLength = experiment === 'length';
  const isModels = isTemp && $('#evidence-sweep').value === 'models';
  const current = data[isModels ? 'temperature_models' : experiment];
  $('#evidence-sweep-label').hidden = !isTemp;
  $('#evidence-encoder-label').hidden = !isTemp && !isLength && !(experiment === 'coverage' && current.directions.dream);
  $('#evidence-length-label').hidden = !isTemp || isModels;
  $('#evidence-generator-label').hidden = !isLength;
  $('#evidence-path-label').hidden = experiment !== 'coverage';
  $('#evidence-mixture').hidden = experiment !== 'coverage';
  $('#evidence-excerpt').hidden = !isTemp || isModels;
  $('#evidence-source').hidden = !isModels;
  $('#evidence-protocol').textContent = current.protocol;
  $('#evidence-uncertainty').textContent = `${current.aggregation} Computed ${current.computed}.`;
  let rows, columns, valueLabel;
  if (isModels) {
    const encoder = $('#evidence-encoder').value;
    const measurements = current.rows.filter(r => r.encoder === encoder);
    const models = Object.entries(current.models);
    rows = [...new Set(measurements.map(r => r.temperature))].sort((a, b) => a - b).map(temperature => ({
      temperature, scores: Object.fromEntries(measurements.filter(r => r.temperature === temperature).map(r => [r.model, r])),
    }));
    const series = models.map(([id, label]) => ({ id, label, value: r => r.scores[id].mean,
      best: rows.reduce((a, b) => a.scores[id].mean < b.scores[id].mean ? a : b).temperature }));
    const bestTemperatures = new Set(series.map(s => s.best));
    const budget = current.directions[encoder].toLocaleString();
    $('#evidence-title').textContent = 'How temperature changes three models.';
    $('#evidence-summary').textContent = bestTemperatures.size === 1
      ? `All three models reach their lowest measured SOL at temperature ${series[0].best.toFixed(1)} under ${names[encoder]}. Select a temperature to compare their distances to reference G.`
      : series.map(s => `${s.label}: lowest SOL at temperature ${s.best.toFixed(1)}`).join('. ') + '.';
    $('#evidence-context').textContent = `${names[encoder]} · ${budget} × ${budget} directions · reference G · 5,000 texts per side · 1,024 tokens`;
    $('#evidence-source').innerHTML = `<a href="./assets/${escape(current.tables[encoder])}" download>Download ${escape(names[encoder])} source table (LaTeX) ↓</a>`;
    chart = { rows, log: true, label: 'SOL distance ↓ · log scale', axis: 'Sampling temperature', xValue: r => r.temperature, xLabel: r => r.temperature.toFixed(1), series };
    columns = ['Temperature', ...models.map(([, label]) => label)];
    valueLabel = r => `Temperature ${r.temperature.toFixed(1)}`;
  } else if (isTemp) {
    const encoder = $('#evidence-encoder').value;
    const length = Number($('#evidence-length').value);
    rows = current.rows.filter(r => r.encoder === encoder && r.length === length);
    const best = rows.reduce((a, b) => a.mean < b.mean ? a : b);
    $('#evidence-title').textContent = 'Temperature changes the distribution.';
    $('#evidence-summary').textContent = `For these saved GPT-2 Large samples, temperature ${best.temperature.toFixed(1)} gives the smallest SOL under ${names[encoder]} at a ${length.toLocaleString()}-token cap.`;
    $('#evidence-context').textContent = `${names[encoder]} encoder · 2,400 texts per corpus · GPT-2 Large generator`;
    chart = { rows, log: true, label: 'SOL distance ↓ · log scale', axis: 'Sampling temperature', xValue: r => r.temperature, xLabel: r => r.temperature.toFixed(1), series: [{ label: 'SOL', value: r => r.mean }] };
    columns = ['Temperature', 'SOL mean', 'Population SD'];
    valueLabel = r => `Temperature ${r.temperature.toFixed(1)}`;
  } else if (isLength) {
    const encoder = $('#evidence-encoder').value;
    const generator = $('#evidence-generator').value;
    rows = current.rows.filter(r => r.encoder === encoder && r.generator === generator).sort((a, b) => a.tokens - b.tokens);
    const change = rows[2].mean - rows[0].mean, budget = current.directions[encoder].toLocaleString();
    $('#evidence-title').textContent = 'The amount of text matters.';
    $('#evidence-summary').textContent = `Under ${names[encoder]}, ${generators[generator]} changes from ${number(rows[0].mean)} at 256 tokens to ${number(rows[2].mean)} at 1,024 tokens (${change < 0 ? '−' : '+'}${number(Math.abs(change))}). Both corpora use the same cap at each point.`;
    $('#evidence-context').textContent = `${names[encoder]} encoder · ${budget} × ${budget} directions · 5,000 texts per corpus · ${generators[generator]} generator`;
    chart = { rows, label: 'SOL distance ↓', axis: 'Token cap on both corpora', xValue: r => r.tokens, xLabel: r => r.tokens.toLocaleString(), series: [{ label: 'SOL', value: r => r.mean }] };
    columns = ['Token cap', 'SOL mean', 'Population SD'];
    valueLabel = r => `${r.tokens.toLocaleString()} tokens`;
  } else {
    // GPT-2 Large is the fallback while a Dream run is not exported.
    const encoder = current.directions[$('#evidence-encoder').value] ? $('#evidence-encoder').value : 'gpt2';
    const path = $('#evidence-path').value;
    const removeSports = path === 'Q2_to_Q1';
    const budget = current.directions[encoder].toLocaleString();
    rows = current.rows.filter(r => r.encoder === encoder && r.path === path).sort((a, b) => a.target_share_percent - b.target_share_percent);
    $('#evidence-title').textContent = removeSports ? 'What happens when Sports disappears?' : 'What happens when new topics appear?';
    $('#evidence-summary').textContent = `The reference stays 50% World and 50% Sports. Under ${names[encoder]}, SOL goes from ${number(rows[0].sol.mean)} for a separate sample of that mix to ${number(rows.at(-1).sol.mean)} ${removeSports ? 'for World only' : 'for an even mix of all four topics'}.`;
    $('#evidence-context').textContent = `AG News · ${names[encoder]} encoder · ${budget} × ${budget} directions · 5,000 articles per corpus · 1,024-token cap`;
    chart = { rows, label: 'SOL distance ↓', axis: 'Starting corpus replaced (%)', xValue: r => r.target_share_percent, xLabel: r => String(r.target_share_percent), showTick: r => r.target_share_percent % 20 === 0, series: [{ label: 'SOL', value: r => r.sol.mean }] };
    columns = ['Replacement share', 'SOL mean ± SD', 'Candidate topic proportions'];
    valueLabel = r => `${r.target_share_percent}% replaced with ${removeSports ? 'World-only articles' : 'an even four-topic mix'}`;
  }
  selected = Math.min(selected, rows.length - 1);
  const row = rows[selected];
  $('#evidence-setting').max = String(rows.length - 1);
  $('#evidence-setting').value = String(selected);
  $('#evidence-setting').disabled = false;
  $('#evidence-setting').setAttribute('aria-valuetext', valueLabel(row));
  $('#evidence-setting-label').textContent = valueLabel(row);
  $('#evidence-selected').classList.toggle('is-comparison', isModels);
  let selectedText;
  if (isModels) {
    selectedText = chart.series.map(s => `${s.label}: SOL ${tableNumber(s.value(row))}`).join('; ');
    $('#evidence-selected').innerHTML = chart.series.map((s, i) => `<span class="evidence-model-value" data-model="${s.id}" style="--series-color:${colors[i]}"><span>${escape(s.label)}</span> <strong>${tableNumber(s.value(row))}</strong></span>`).join('\n');
  } else {
    selectedText = experiment === 'coverage'
      ? `SOL ${number(row.sol.mean)} ± ${number(row.sol.std)}`
      : `SOL ${number(row.mean)} ± ${number(row.std)}`;
    $('#evidence-selected').textContent = selectedText;
  }
  $('#evidence-chart').tabIndex = 0;
  $('#evidence-chart').setAttribute('aria-disabled', 'false');
  $('#evidence-chart').setAttribute('aria-valuemax', String(rows.length - 1));
  $('#evidence-chart').setAttribute('aria-valuenow', String(selected));
  $('#evidence-chart').setAttribute('aria-valuetext', `${valueLabel(row)}. ${selectedText}`);
  if (isTemp && !isModels) $('#evidence-text').textContent = current.excerpts[String(row.temperature)] || current.excerpts[row.temperature.toFixed(1)];
  if (experiment === 'coverage') $('#evidence-mixture').textContent = `Candidate: ${topicMixture(row)}`;
  $('#evidence-legend').innerHTML = chart.series.map((s, i) => `<span><i style="border-top:3px ${['solid', 'dashed', 'dotted'][i]} ${colors[i]}"></i>${escape(s.label)}</span>`).join('');
  $('#evidence-table-caption').textContent = isModels
    ? `${names[$('#evidence-encoder').value]}: SOL mean ± population SD over nine seed pairs. Bold marks each model’s minimum.`
    : `${$('#evidence-title').textContent} Raw distances and population standard deviations.`;
  $('#evidence-table-head').innerHTML = `<tr>${columns.map(c => `<th scope="col">${escape(c)}</th>`).join('')}</tr>`;
  $('#evidence-table-body').innerHTML = isModels
    ? rows.map(r => `<tr><td>${r.temperature.toFixed(1)}</td>${chart.series.map(s => {
      const score = r.scores[s.id];
      const value = `${tableNumber(score.mean)} ± ${tableNumber(score.std)}`;
      return `<td>${r.temperature === s.best ? `<strong>${value}</strong>` : value}</td>`;
    }).join('')}</tr>`).join('')
    : rows.map(r => `<tr>${(experiment === 'coverage'
      ? [`${r.target_share_percent}%`, `${number(r.sol.mean)} ± ${number(r.sol.std)}`, topicMixture(r)]
      : [valueLabel(r), number(r.mean), number(r.std)]).map(v => `<td>${escape(v)}</td>`).join('')}</tr>`).join('');
  drawChart();
}

function selectExperiment(key) {
  experiment = key;
  selected = key === 'temperature' ? ($('#evidence-sweep').value === 'models' ? 3 : 2) : 0;
  document.querySelectorAll('[data-experiment]').forEach(button => {
    const active = button.dataset.experiment === key;
    button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
  });
  $('#evidence-panel').setAttribute('aria-labelledby', `example-${key}`);
  render();
}

function selectSetting(index) {
  if (!chart) return;
  const next = Math.max(0, Math.min(index, chart.rows.length - 1));
  if (next === selected) return;
  selected = next;
  render();
}

function plotPoint(event) {
  const svg = $('#evidence-chart svg');
  const transform = svg?.getScreenCTM();
  if (!transform) return null;
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(transform.inverse());
}

function selectPlotPoint(point) {
  const nearest = chart.positions.reduce((best, x, index, positions) =>
    Math.abs(x - point.x) < Math.abs(positions[best] - point.x) ? index : best, 0);
  selectSetting(nearest);
}

const plot = $('#evidence-chart');
plot.addEventListener('pointerdown', event => {
  if (!chart || !event.isPrimary || event.button !== 0) return;
  const point = plotPoint(event);
  const { left, right, top, bottom } = chart.bounds;
  if (!point || point.x < left - 12 || point.x > right + 12 || point.y < top - 12 || point.y > bottom + 30) return;
  // Capture on the container: its SVG is replaced whenever the setting changes.
  dragPointer = event.pointerId;
  plot.setPointerCapture(event.pointerId);
  plot.focus({ preventScroll: true });
  selectPlotPoint(point);
});
plot.addEventListener('pointermove', event => {
  if (event.pointerId !== dragPointer) return;
  const point = plotPoint(event);
  if (point) selectPlotPoint(point);
});
plot.addEventListener('pointerup', event => {
  if (event.pointerId !== dragPointer) return;
  const point = plotPoint(event);
  if (point) selectPlotPoint(point);
  dragPointer = null;
  if (plot.hasPointerCapture(event.pointerId)) plot.releasePointerCapture(event.pointerId);
});
for (const type of ['pointercancel', 'lostpointercapture']) {
  plot.addEventListener(type, event => { if (event.pointerId === dragPointer) dragPointer = null; });
}
plot.addEventListener('keydown', event => {
  if (!chart || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  selectSetting(event.key === 'Home' ? 0 : event.key === 'End' ? chart.rows.length - 1
    : selected + (['ArrowRight', 'ArrowUp'].includes(event.key) ? 1 : -1));
});
document.querySelectorAll('[data-experiment]').forEach(button => button.addEventListener('click', () => selectExperiment(button.dataset.experiment)));
$('.evidence-tabs').addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const keys = ['temperature', 'length', 'coverage'];
  const index = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (keys.indexOf(experiment) + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
  selectExperiment(keys[index]); $(`#example-${keys[index]}`).focus();
});
['encoder', 'length', 'generator', 'path'].forEach(key => $(`#evidence-${key}`).addEventListener('change', render));
$('#evidence-sweep').addEventListener('change', () => { selected = $('#evidence-sweep').value === 'models' ? 3 : 2; render(); });
$('#evidence-setting').addEventListener('input', event => selectSetting(Number(event.target.value)));
new ResizeObserver(drawChart).observe($('#evidence-chart'));
async function loadEvidence() {
  try {
    const response = await fetch('./assets/evidence.json');
    if (!response.ok) throw new Error('Unavailable');
    data = await response.json(); render();
  } catch {
    $('#evidence-title').textContent = 'Measured examples could not load.';
    $('#evidence-summary').innerHTML = '<button id="retry-evidence" class="text-button">Try again</button> or <a href="./assets/evidence.csv">download the values</a>.';
    $('#retry-evidence').addEventListener('click', loadEvidence);
  }
}
loadEvidence();
