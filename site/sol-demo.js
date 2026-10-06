// Small deterministic double-sliced Wasserstein example on synthetic 2D clouds.
// Uses empirical bin-average quantiles and raw RBF Gaussian-process directions.
// No transformer runs in the browser; these coordinates are illustrative.
export function random(seed = 17) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

function normal(rng) {
  return Math.sqrt(-2 * Math.log(rng())) * Math.cos(2 * Math.PI * rng());
}

export function makeCorpora(shift = 0.35, scenario = 'shift') {
  const rng = random(79);
  const centers = [[-1.1, -.65], [-.1, .75], [.9, -.25], [-.6, .1], [.75, .9], [.2, -.95]];
  const reference = centers.map(([cx, cy]) => Array.from({ length: 12 }, () => [cx + normal(rng) * .23, cy + normal(rng) * .23]));
  const generated = reference.map((cloud, d) => cloud.map(([x, y], t) => {
    if (scenario === 'match') return [x, y];
    if (scenario === 'collapse') return [...reference[0][t]];
    if (scenario === 'spread') {
      const mean = cloud.reduce((sum, point) => [sum[0] + point[0] / cloud.length, sum[1] + point[1] / cloud.length], [0, 0]);
      return [mean[0] + .1 * (x - mean[0]), mean[1] + .1 * (y - mean[1])];
    }
    return [x + shift * 1.65, y + shift * (.65 + Math.sin(d * 1.7) * .3)];
  }));
  return { reference, generated };
}

export function binQuantiles(values, count) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return Array.from({ length: count }, (_, m) => {
    let sum = 0;
    const start = m / count, end = (m + 1) / count;
    for (let i = Math.floor(start * n); i < Math.min(n, Math.ceil(end * n)); i++) {
      sum += sorted[i] * Math.max(0, Math.min(end, (i + 1) / n) - Math.max(start, i / n));
    }
    return sum * count;
  });
}

const M = 24, L = 32, K = 24;
// SOL scales DSW² by the hidden dimension; the toy token states live in ℝ².
const D = 2;
function makeDirections() {
  const rng = random(2026);
  const chol = Array.from({ length: M }, () => Array(M).fill(0));
  for (let i = 0; i < M; i++) {
    for (let j = 0; j <= i; j++) {
      const d = (i - j) / M;
      let s = Math.exp(-d * d / (2 * .1 ** 2)) + (i === j ? 1e-6 : 0);
      for (let k = 0; k < j; k++) s -= chol[i][k] * chol[j][k];
      chol[i][j] = i === j ? Math.sqrt(Math.max(0, s)) : s / chol[j][j];
    }
  }
  const gp = Array.from({ length: K }, () => {
    const z = Array.from({ length: M }, () => normal(rng));
    return chol.map(row => row.reduce((sum, c, j) => sum + c * z[j], 0));
  });
  const token = Array.from({ length: L }, () => {
    const angle = rng() * Math.PI * 2;
    return [Math.cos(angle), Math.sin(angle)];
  });
  return { gp, token };
}
const directions = makeDirections();

export function sliceCorpora(reference, generated, index = 0, gpIndex = 0) {
  const direction = directions.token[index % L];
  const gp = directions.gp[gpIndex % K];
  const project = corpus => {
    const tokens = corpus.map(cloud => cloud.map(([x, y]) => x * direction[0] + y * direction[1]).sort((a, b) => a - b));
    const quantiles = tokens.map(points => binQuantiles(points, M));
    const sequenceValues = quantiles.map(q => q.reduce((sum, v, m) => sum + v * gp[m], 0) / M);
    const ranked = sequenceValues.map((value, sequence) => ({ sequence, value })).sort((a, b) => a.value - b.value || a.sequence - b.sequence);
    return { tokens, quantiles, sequenceValues, ranked, values: ranked.map(row => row.value) };
  };
  return { reference: project(reference), generated: project(generated), direction, gp };
}

export function toySOL(reference, generated) {
  if (!reference.length || reference.length !== generated.length) throw new Error('Toy demo requires equally sized, nonempty corpora.');
  let total = 0;
  for (let l = 0; l < L; l++) {
    const u = directions.token[l];
    const quantiles = corpus => corpus.map(cloud => binQuantiles(cloud.map(([x, y]) => x * u[0] + y * u[1]), M));
    const a = quantiles(reference), b = quantiles(generated);
    for (const gp of directions.gp) {
      const project = qs => qs.map(q => q.reduce((sum, v, m) => sum + v * gp[m], 0) / M).sort((x, y) => x - y);
      const pa = project(a), pb = project(b);
      for (let i = 0; i < pa.length; i++) total += (pa[i] - pb[i]) ** 2 / pa.length;
    }
  }
  return Math.sqrt(D * total / (L * K));
}
