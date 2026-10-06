// SOL ranks span all selected references before search and family filtering.
export const PAGE_SIZE = 18;
export const DEFAULT_FILTERS = { reference: 'all', encoder: 'dream', family: 'all', search: '', sort: 'sol', ascending: true, page: 0 };

export function selectLeaderboard(data, state) {
  const references = state.reference === 'all' ? ['G', 'E', 'B'] : [state.reference];
  const ranked = data.rows.filter(row => references.includes(row.reference) && row.family !== 'reference')
    .sort((a, b) => a.scores[state.encoder].mean - b.scores[state.encoder].mean || a.id.localeCompare(b.id));
  const ranks = new Map(ranked.map((row, i) => [row.id, i + 1]));
  const search = state.search.trim().toLowerCase();
  const allRows = ranked.filter(row => {
    const family = state.family === 'all' || row.family === state.family;
    return family && `${row.model} ${row.setting}`.toLowerCase().includes(search);
  });
  // Sorting by another column preserves each model's SOL rank.
  allRows.sort((a, b) => {
    const value = row => state.sort === 'sol' ? row.scores[state.encoder].mean : row[state.sort];
    return (value(a) - value(b)) * (state.ascending ? 1 : -1) || a.id.localeCompare(b.id);
  });
  const pages = Math.max(1, Math.ceil(allRows.length / PAGE_SIZE));
  const page = Math.max(0, Math.min(state.page, pages - 1));
  const start = page * PAGE_SIZE;
  return { ranks, pages, page, start, total: allRows.length, allRows,
    rows: allRows.slice(start, start + PAGE_SIZE),
    floors: data.rows.filter(row => references.includes(row.reference) && row.family === 'reference') };
}
