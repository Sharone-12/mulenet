export const API = 'http://localhost:8000';
export const WS = 'ws://localhost:8000/ws/chat';

export const C = {
  paper: '#F0EBE5',
  card: '#FBF8F5',
  ink: '#0A0A0A',
  muted: '#6B655E',
  line: '#0A0A0A',

  blue: '#1668E3',
  amber: '#FFB400',
  orange: '#FF5B22',
  green: '#00A94F',
  purple: '#A855F7',

  // Graph semantics. Controllers read hottest, mules warn, ordinary
  // accounts recede into the paper rather than competing with them.
  // Block fills, which carry black text on top. Bright is correct here.
  controller: '#E8412F',
  mule: '#FFB400',
  clean: '#BDB4A8',

  // Small MARKS drawn on the cream surface. The block fills fail contrast
  // badly as marks (amber 1.5:1, clean 1.73:1 against #F0EBE5) and would be
  // near-invisible as graph nodes; these pass >= 3:1.
  //
  // Colour alone still cannot separate controller from mule - darkened red
  // and amber measure dE 0.6-4.3 under deuteranopia - so every mark also
  // encodes role by shape and size.
  markController: '#D2351F',
  markMule: '#B87400',
  markClean: '#8A8177',
};

export const VIEWS = [
  { id: 'network', n: '01', label: 'Network', color: C.blue },
  { id: 'rings', n: '02', label: 'Ring Queue', color: C.amber },
  { id: 'inspect', n: '03', label: 'Investigate', color: C.orange },
  { id: 'benchmark', n: '04', label: 'Benchmark', color: C.green },
];

// Graph nodes are marks on the cream surface, so they use the validated
// mark variants rather than the block fills.
export const roleColor = (role) =>
  role === 'controller' ? C.markController : role === 'mule' ? C.markMule : C.markClean;

export const roleSize = (role) =>
  role === 'controller' ? 9 : role === 'mule' ? 5 : 1.6;

/** Amounts arrive per currency and are never summed across them. */
export function formatAmounts(byCurrency, { compact = false } = {}) {
  if (!byCurrency) return '—';
  const entries = Object.entries(byCurrency);
  if (!entries.length) return '—';
  const fmt = (v) =>
    v.toLocaleString(undefined, {
      maximumFractionDigits: compact && v > 9999 ? 0 : 2,
      notation: compact && v > 999999 ? 'compact' : 'standard',
    });
  return entries.map(([cur, amt]) => `${fmt(amt)} ${cur}`).join(' · ');
}

export const num = (v) => (v ?? 0).toLocaleString();
