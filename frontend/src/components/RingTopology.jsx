import { useMemo } from 'react';
import { C } from '../theme';

/**
 * Deterministic node-link diagram of one ring.
 *
 * A force simulation is the wrong tool here: rings are 2-8 accounts, and a
 * physics layout renders them differently on every mount, which reads as
 * noise. Laying them out by their position in the flow instead means the
 * shape itself is the finding - a fan, a chain and a cycle are distinguishable
 * at a glance.
 *
 * Role is encoded by SHAPE as well as colour (controller = square,
 * mule = circle). Dark red and dark amber are near-identical under
 * deuteranopia (measured dE 0.6-4.3), so colour alone cannot carry it.
 */
export default function RingTopology({ ring, roles, compact = false, height }) {
  const layout = useMemo(() => {
    if (!ring?.accounts?.length) return null;

    const accounts = ring.accounts;
    const links = (ring.transactions ?? []).map((t) => ({ from: t.from, to: t.to }));

    const sources = new Set(ring.source_accounts ?? []);
    const cashouts = new Set(ring.cashout_accounts ?? []);
    const middle = accounts.filter((a) => !sources.has(a) && !cashouts.has(a));

    // A ring with no entry or exit point is a closed loop; a column layout
    // would hide exactly the property that makes it suspicious.
    const isCycle = sources.size === 0 && cashouts.size === 0;

    const w = 100;
    const h = 100;
    const pos = {};

    if (isCycle) {
      const r = accounts.length === 2 ? 26 : 32;
      accounts.forEach((a, i) => {
        const angle = (i / accounts.length) * Math.PI * 2 - Math.PI / 2;
        pos[a] = { x: 50 + r * Math.cos(angle), y: 50 + r * Math.sin(angle) };
      });
    } else {
      const columns = [[...sources], middle, [...cashouts]].filter((c) => c.length);
      const gap = columns.length === 1 ? 0 : (w - 24) / (columns.length - 1);
      columns.forEach((col, ci) => {
        const x = columns.length === 1 ? 50 : 12 + ci * gap;
        col.forEach((a, ri) => {
          const spread = col.length === 1 ? 0 : (h - 34) / (col.length - 1);
          const y = col.length === 1 ? 50 : 17 + ri * spread;
          pos[a] = { x, y };
        });
      });
    }

    return { pos, links, accounts, isCycle };
  }, [ring]);

  if (!layout) return null;

  const roleOf = (a) => {
    if (roles?.controllers?.some((c) => c.account === a)) return 'controller';
    if (roles?.mules?.some((m) => m.account === a)) return 'mule';
    return 'clean';
  };

  const markColor = (role) =>
    role === 'controller' ? C.markController : role === 'mule' ? C.markMule : C.markClean;

  const r = compact ? 4.2 : 5.6;
  const showLabels = !compact;

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="xMidYMid meet"
      style={{ width: '100%', height: height ?? (compact ? 74 : 210), display: 'block' }}
      role="img"
      aria-label={`Ring ${ring.ring_id ?? ''} structure: ${layout.accounts.length} accounts`}
    >
      <defs>
        <marker
          id={`arw-${compact ? 'c' : 'f'}`}
          markerWidth="5"
          markerHeight="5"
          refX="4.4"
          refY="2"
          orient="auto"
        >
          <path d="M0,0 L4.4,2 L0,4 Z" fill="rgba(10,10,10,0.42)" />
        </marker>
      </defs>

      {layout.links.map((l, i) => {
        const a = layout.pos[l.from];
        const b = layout.pos[l.to];
        if (!a || !b || (a.x === b.x && a.y === b.y)) return null;
        // Bow the path so parallel transfers between the same pair stay
        // individually visible rather than stacking into one line.
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2 - 7 - i * 1.6;
        return (
          <path
            key={i}
            d={`M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`}
            fill="none"
            stroke="rgba(10,10,10,0.42)"
            strokeWidth={compact ? 0.7 : 0.9}
            markerEnd={`url(#arw-${compact ? 'c' : 'f'})`}
          />
        );
      })}

      {layout.accounts.map((a) => {
        const p = layout.pos[a];
        if (!p) return null;
        const role = roleOf(a);
        const fill = markColor(role);
        return (
          <g key={a}>
            {role === 'controller' ? (
              <rect
                x={p.x - r}
                y={p.y - r}
                width={r * 2}
                height={r * 2}
                rx={1.4}
                fill={fill}
                stroke={C.paper}
                strokeWidth="1.1"
              />
            ) : (
              <circle
                cx={p.x}
                cy={p.y}
                r={role === 'mule' ? r * 0.82 : r * 0.55}
                fill={fill}
                stroke={C.paper}
                strokeWidth="1.1"
              />
            )}
            {showLabels && (
              <text
                x={p.x}
                y={p.y + r + 6.5}
                textAnchor="middle"
                fontSize="3.5"
                fontWeight="600"
                fill="rgba(10,10,10,0.72)"
              >
                {a.split('-')[1]?.slice(-6) ?? a}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
