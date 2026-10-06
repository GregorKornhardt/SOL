// Text and MathML formulas for the four method stages.
// Stage formulas use native MathML.
const mi = x => `<mi>${x}</mi>`, mn = x => `<mn>${x}</mn>`;
// Fences keep their text size; display-style integrals would otherwise stretch them.
const mo = x => ('()[]⟨⟩'.includes(x) ? `<mo stretchy="false">${x}</mo>` : `<mo>${x}</mo>`);
const sub = (base, script) => `<msub>${base}<mrow>${script}</mrow></msub>`;
const sup = (base, script) => `<msup>${base}<mrow>${script}</mrow></msup>`;
const math = body => `<math display="block">${body}</math>`;
const sphere = sup(mi('𝕊'), mi('d') + mo('−') + mn(1));
const sliceMap = sub(mi('M'), mi('ξ') + mo(',') + mi('g'));
const pushforward = corpus => sub(`<mrow>${mo('(') + sliceMap + mo(')')}</mrow>`, mo('♯')) + mi(corpus);
const dsw2 = sup(mi('DSW'), mn(2));
export const formulas = [
  math(`<mtext>text</mtext>${mo('→')}${sub(mi('h'), mn(1))}${mo(',')}${mo('…')}${mo(',')}${sub(mi('h'), mi('n'))}${mo('∈')}${sup(mi('ℝ'), mi('d'))}`),
  math(`${sub(mi('h'), mi('i'))}${mo('↦')}${mo('⟨')}${mi('ξ')}${mo(',')}${sub(mi('h'), mi('i'))}${mo('⟩')}${mo(',')}<mspace width="1em"/>${mi('ξ')}${mo('∈')}${sphere}`),
  math(`${sliceMap}${mo('(')}${sub(mi('μ'), mi('i'))}${mo(')')}${mo('=')}<msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup>${mi('Q')}${mo('(')}${mi('u')}${mo(')')}<mspace width="0.17em"/>${mi('g')}${mo('(')}${mi('u')}${mo(')')}<mspace width="0.17em"/><mi mathvariant="normal">d</mi>${mi('u')}`),
  sol(),
];
// Step 4 uses one line for DSW² when the text column is wide enough, else two.
function sol() {
  const expectation = sub(mi('𝔼'), mi('ξ') + mo('∼') + mi('𝒰') + mo('(') + sphere + mo(')') + mo(',') + mi('g') + mo('∼') + mi('γ'));
  const transport = `${mo('[')}<msubsup><mi mathvariant="normal">W</mi><mn>2</mn><mn>2</mn></msubsup>${mo('(')}${pushforward('μ')}${mo(',')}${pushforward('ν')}${mo(')')}${mo(']')}`;
  const root = math(`${mi('SOL')}${mo('=')}<msqrt>${mi('d')}${mo('⋅')}${dsw2}</msqrt>`);
  return `<span class="formula-wide">${math(dsw2 + mo('=') + expectation + transport)}${root}</span>`
    + `<span class="formula-narrow">${math(dsw2 + mo('=') + expectation)}${math('<mspace width="2.2em"/>' + transport)}${root}</span>`;
}
export const stages = [
  { title: 'Watch each text become a token cloud.', text: 'Start with six different text sequences in each corpus. The encoder embeds every token into its hidden space ℝᵈ, so each sequence becomes a cloud of d-dimensional token states. Here d = 2: play the mapping to watch the tokens move into one shared space, with one labeled cloud for each sequence.', formula: formulas[0], note: 'One dot is a token state; the whole labeled cloud represents its sequence. Both corpora keep all six sequences through the two slices.', readout: '12 SEQUENCES TOTAL · 12 TOKENS EACH', caption: 'Illustrative sentences with synthetic coordinates in ℝ². The benchmark encoders use d = 1,280 (GPT-2 Large) and d = 3,584 (Dream 7B).' },
  { title: 'Project each sequence’s tokens.', text: 'We project every token vector onto the same line, defined by the direction ξ. Each token is then represented by a single number: its position along that line. For each sequence, these values form a one-dimensional distribution, shown in a separate row.', formula: formulas[1], note: 'All 12 sequences use the same ξ. The rows share one scale: R1–R6 above, G1–G6 below. Select a sequence to follow its tokens.', readout: 'ONE TOKEN DIRECTION · 12 SEQUENCES', caption: 'The same clouds from step 1 project onto ξ, then form 12 separate token distributions. Each dot keeps its sequence and projected value.' },
  { title: 'One curve, then one value per sequence.', text: 'The sorted token values define a quantile curve Q: at each fraction u, it gives the corresponding token value. Integrating Q against the same Gaussian-process function g gives one scalar per sequence.', formula: formulas[2], note: 'Each curve belongs to one sequence μᵢ, and M turns it into one number. After this second slice, each corpus has six scalar values—one for each of its six sequences.', readout: '12 QUANTILE CURVES → 12 SCALARS', caption: 'The curves share one set of axes and the same function g(u).' },
  { title: 'Compare the two collections of sequences.', text: 'Each pair of directions (ξ, g) leaves a one-dimensional optimal transport problem between the six scalars of each corpus. In one dimension it is solved exactly by sorting: match the values by rank and take their squared differences, in O(n log n) time for n sequences. Average these costs over all token and GP directions, multiply by the hidden dimension d, then take one square root to obtain the corpus distance.', formula: formulas[3], note: 'μ and ν are the reference and generated corpora. The lines join sorted values, so rank 1 does not necessarily mean sequence R1 or G1. This synthetic score averages 32 × 24 slice pairs with d = 2.', readout: '768 SLICE PAIRS · ONE CORPUS DISTANCE', caption: 'One dot per sequence. Lines match equal ranks between the two scalar distributions.' },
];
