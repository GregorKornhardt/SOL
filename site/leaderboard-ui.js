// OpenWebText leaderboard: filters, sorting, paging, details, and CSV export.
import { DEFAULT_FILTERS, PAGE_SIZE, selectLeaderboard } from './leaderboard.js';
import { $, $$, escapeHTML, pad } from './dom.js';
import { toast } from './site-chrome.js';

// Leaderboard entries come from the repository's SOL result table.
const encoderNames = { dream: 'Dream 7B', gpt2: 'GPT-2 Large' };
const familyNames = { autoregressive: 'AUTOREGRESSIVE', discrete: 'DISCRETE DIFF.', continuous: 'CONTINUOUS DIFF.', fewstep: 'FEW-STEP' };
const state = { ...DEFAULT_FILTERS };
const referenceNames = { G: 'GPT-2-packed OpenWebText', E: 'ELF OpenWebText', B: 'BD3-packed OpenWebText' };
let benchmark = null;
let filteredRows = [];
function formatScore(value) { return value < 1 ? value.toFixed(4) : value.toFixed(3); }

function renderLeaderboard() {
  if (!benchmark) return;
  const { ranks, pages, page, start, total, allRows, rows, floors } = selectLeaderboard(benchmark, state);
  state.page = page;
  filteredRows = allRows;
  $('#leaderboard-body').innerHTML = total ? rows.map(row => {
      const score = row.scores[state.encoder], rank = ranks.get(row.id);
      return `<tr data-reference="${row.reference}" class="${rank === 1 ? 'first-place' : ''}"><td class="rank-col">${pad(rank)}</td><td class="model-col"><div class="model-line"><button class="model-button" data-detail="${row.id}">${escapeHTML(row.model)}</button><a class="reference-badge reference-${row.reference.toLowerCase()}" href="#reference-key" aria-label="Reference ${row.reference}: ${referenceNames[row.reference]}" title="Reference ${row.reference}: ${referenceNames[row.reference]}">${row.reference}</a></div><span class="model-setting">${escapeHTML(row.setting)}</span></td><td class="family-col"><span class="family-tag ${row.family}">${familyNames[row.family]}</span></td><td class="score-col"><span class="score-value">${formatScore(score.mean)}</span><span class="score-std">± ${formatScore(score.std)}</span></td><td class="ppl-col">${row.ppl.toFixed(1)}</td><td class="entropy-col">${row.entropy.toFixed(3)}</td><td class="detail-col"><button class="detail-arrow" data-detail="${row.id}" aria-label="Details for ${escapeHTML(row.model)}, reference ${row.reference}: ${escapeHTML(row.setting)}">↗</button></td></tr>`;
    }).join('') : '<tr><td colspan="7" class="table-message">No matching configurations.<br>Try another model family, reference, or search term.</td></tr>';
  $('#reference-floor').innerHTML = floors.map(floor => {
    const score = floor.scores[state.encoder];
    return `<div><span>◇ ${floor.reference} · REAL–REAL CONTROL</span><strong>SOL ${formatScore(score.mean)} ± ${formatScore(score.std)}</strong></div>`;
  }).join('');
  $('#board-context').textContent = `${encoderNames[state.encoder]} encoder · ${state.reference === 'all' ? 'All references in one SOL ranking' : `${referenceNames[state.reference]} reference`} · ${PAGE_SIZE} results per page`;
  $('#results-count').textContent = total ? `${pad(start + 1)}–${pad(Math.min(start + PAGE_SIZE, total))} of ${total} configurations` : '0 matching configurations';
  $('#page-count').textContent = `${pad(state.page + 1)} / ${pad(pages)}`;
  $('#prev-page').disabled = state.page === 0;
  $('#next-page').disabled = state.page >= pages - 1;
  $('#export-csv').disabled = !filteredRows.length;
  $$('[data-sort]').forEach(button => {
    const active = button.dataset.sort === state.sort;
    if (active) button.parentElement.setAttribute('aria-sort', state.ascending ? 'ascending' : 'descending');
    else button.parentElement.removeAttribute('aria-sort');
    button.querySelector('span').textContent = active ? state.ascending ? '↓' : '↑' : '↕';
  });
}

async function loadLeaderboard() {
  try {
    const response = await fetch('./assets/leaderboard.json');
    if (!response.ok) throw new Error('Could not load data');
    benchmark = await response.json();
    renderLeaderboard();
  } catch {
    $('#leaderboard-body').innerHTML = '<tr><td colspan="7" class="table-message">The results could not be loaded. <button class="text-button" id="retry-data">Try again ↻</button> or <a href="./assets/benchmark-source.tex">open the source table ↗</a>.</td></tr>';
    $('#results-count').textContent = 'Benchmark data unavailable';
    $('#export-csv').disabled = true;
    $('#retry-data').addEventListener('click', loadLeaderboard);
  }
}

export function initLeaderboard() {
  loadLeaderboard();
  $$('[data-family]').forEach(button => button.addEventListener('click', () => {
    state.family = button.dataset.family; state.page = 0;
    $$('[data-family]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
    renderLeaderboard();
  }));
  ['reference', 'encoder'].forEach(key => $(`#${key}`).addEventListener('change', event => { state[key] = event.target.value; state.page = 0; renderLeaderboard(); }));
  $('#model-search').addEventListener('input', event => { state.search = event.target.value; state.page = 0; renderLeaderboard(); });
  $$('[data-sort]').forEach(button => button.addEventListener('click', () => {
    const key = button.dataset.sort;
    state.ascending = key === state.sort ? !state.ascending : true;
    state.sort = key; state.page = 0; renderLeaderboard();
  }));
  $('#prev-page').addEventListener('click', () => { state.page--; renderLeaderboard(); });
  $('#next-page').addEventListener('click', () => { state.page++; renderLeaderboard(); });
  $('#leaderboard-body').addEventListener('click', event => {
    const button = event.target.closest('[data-detail]');
    if (!button || !benchmark) return;
    const row = benchmark.rows.find(row => row.id === button.dataset.detail);
    $('#detail-title').textContent = row.model;
    $('#detail-content').innerHTML = `<p class="detail-setting">${escapeHTML(row.setting)}</p><div class="detail-metrics">${Object.keys(encoderNames).map(key => { const score = row.scores[key]; return `<div><span>${encoderNames[key]}</span><strong>${formatScore(score.mean)}</strong><small>± ${formatScore(score.std)}</small></div>`; }).join('')}</div><p>Each column uses a different encoder. Compare models within an encoder; the raw SOL scales differ.</p><dl class="detail-list"><dt>Model family</dt><dd>${familyNames[row.family]}</dd><dt>Reference family</dt><dd>${row.reference} · ${referenceNames[row.reference]}</dd><dt>Samples per corpus</dt><dd>5,000</dd><dt>Token horizon</dt><dd>1,024</dd><dt>Direction / GP seed pairs</dt><dd>9</dd><dt>Quantile bins</dt><dd>64</dd><dt>GPT-2 directions</dt><dd>1,024 × 1,024</dd><dt>Dream directions</dt><dd>8,192 × 8,192</dd><dt>Gen. perplexity</dt><dd>${row.ppl.toFixed(1)}</dd><dt>Token entropy (nats)</dt><dd>${row.entropy.toFixed(3)}</dd></dl><p>The SOL values are means ± population standard deviations. The real–real control estimates the finite-sample floor for the selected reference.</p><a class="dialog-source" href="./assets/benchmark-source.tex" target="_blank" rel="noreferrer">View recorded source data ↗</a>`;
    $('#detail-dialog').showModal();
  });
  $('#export-csv').addEventListener('click', () => {
    if (!filteredRows.length) return;
    const header = ['model', 'setting', 'family', 'reference', 'encoder', 'sol_mean', 'sol_std', 'gen_ppl', 'entropy_nats', 'samples_per_corpus', 'token_horizon', 'seed_pairs', 'source'];
    const rows = filteredRows.map(row => [row.model, row.setting, row.family, row.reference, encoderNames[state.encoder], row.scores[state.encoder].mean, row.scores[state.encoder].std, row.ppl, row.entropy, benchmark.samples, benchmark.tokens, benchmark.seedPairs, benchmark.source]);
    const csv = [header, ...rows].map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n') + '\r\n';
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url; link.download = `sol-owt-${state.reference}-${state.encoder}.csv`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(`Exported ${rows.length} configurations`);
  });
}
