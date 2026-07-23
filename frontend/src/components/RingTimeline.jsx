import { useMemo, useState } from 'react';
import { C } from '../theme';

/**
 * When the money moved, as a lollipop per transaction on a time axis.
 *
 * Timing is the criminological signal in these rings - a chain that completes
 * in 34 minutes reads very differently from the same chain over nine days -
 * and a table of timestamps does not show it. One series, so no legend; the
 * heading names it.
 */
export default function RingTimeline({ ring, height = 130 }) {
  const [hover, setHover] = useState(null);

  const model = useMemo(() => {
    const txns = ring?.transactions ?? [];
    if (txns.length < 2) return null;

    const points = txns
      .map((t) => ({ ...t, t: new Date(t.timestamp.replace(' ', 'T')).getTime() }))
      .filter((p) => Number.isFinite(p.t))
      .sort((a, b) => a.t - b.t);
    if (points.length < 2) return null;

    const t0 = points[0].t;
    const t1 = points[points.length - 1].t;
    const span = Math.max(1, t1 - t0);
    const maxAmt = Math.max(...points.map((p) => p.amount));

    return { points, t0, t1, span, maxAmt };
  }, [ring]);

  if (!model) return null;

  const { points, t0, span, maxAmt } = model;
  const padX = 4;
  const plotH = 68;

  const spanHours = span / 3_600_000;
  const spanLabel =
    spanHours < 48 ? `${spanHours.toFixed(1)} hours` : `${(spanHours / 24).toFixed(1)} days`;

  return (
    <div className="timeline">
      <div className="tl-head">
        <span className="tl-title">Movement over time</span>
        <span className="tl-span">{points.length} transfers across {spanLabel}</span>
      </div>

      <div className="tl-plot" style={{ height }}>
        <svg viewBox={`0 0 100 ${plotH + 16}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
          {/* Baseline: recessive, the marks carry the data. */}
          <line x1="0" y1={plotH} x2="100" y2={plotH} stroke="rgba(10,10,10,0.22)" strokeWidth="0.4" />

          {points.map((p, i) => {
            const x = padX + ((p.t - t0) / span) * (100 - padX * 2);
            const h = Math.max(3, (p.amount / maxAmt) * (plotH - 10));
            const active = hover === i;
            return (
              <g key={i}>
                <line
                  x1={x}
                  y1={plotH}
                  x2={x}
                  y2={plotH - h}
                  stroke={active ? C.ink : C.markController}
                  strokeWidth="1.1"
                  strokeLinecap="round"
                />
                <circle cx={x} cy={plotH - h} r="1.9" fill={active ? C.ink : C.markController} />
                {/* Hit target far larger than the mark. */}
                <rect
                  x={x - 3}
                  y={0}
                  width="6"
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                />
              </g>
            );
          })}
        </svg>

        {hover != null && (
          <div className="tl-tip" style={{ left: `${padX + ((points[hover].t - t0) / span) * (100 - padX * 2)}%` }}>
            <strong>{points[hover].amount.toLocaleString()} {points[hover].currency}</strong>
            <span>{points[hover].timestamp}</span>
            <span>{points[hover].from.split('-')[1]} → {points[hover].to.split('-')[1]}</span>
          </div>
        )}
      </div>

      <div className="tl-axis">
        <span>{points[0].timestamp}</span>
        <span>{points[points.length - 1].timestamp}</span>
      </div>
    </div>
  );
}
