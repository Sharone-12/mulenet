import { C } from '../theme';

/**
 * MuleNet mark: a source account fanning to three mules and reconverging at a
 * cash-out — the fan-out/fan-in ring the detector is built around. So the
 * glyph is the thing the product finds, not decoration.
 *
 * Edges and mule nodes use currentColor so the mark inherits ink in either
 * theme; the two flow endpoints are drawn as controller-red squares, matching
 * the square-vs-circle role encoding used everywhere else in the app.
 */
export default function Logo({ size = 34 }) {
  const s = size;
  return (
    <svg
      width={s}
      height={s}
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label="MuleNet"
      style={{ display: 'block', flexShrink: 0 }}
    >
      {/* edges: source -> mules -> cash-out */}
      <g stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.55">
        <path d="M6 16 L16 7" />
        <path d="M6 16 L16 16" />
        <path d="M6 16 L16 25" />
        <path d="M16 7 L26 16" />
        <path d="M16 16 L26 16" />
        <path d="M16 25 L26 16" />
      </g>

      {/* mules — circles, ink */}
      <g fill="currentColor">
        <circle cx="16" cy="7" r="2.3" />
        <circle cx="16" cy="16" r="2.3" />
        <circle cx="16" cy="25" r="2.3" />
      </g>

      {/* controllers — squares, flagged red, ringed in the surface colour
          (CSS var so the cut-out tracks light/dark, not the static token) */}
      <g fill={C.controller} stroke="var(--paper)" strokeWidth="1.2">
        <rect x="2.4" y="12.4" width="7.2" height="7.2" rx="1.4" />
        <rect x="22.4" y="12.4" width="7.2" height="7.2" rx="1.4" />
      </g>
    </svg>
  );
}
