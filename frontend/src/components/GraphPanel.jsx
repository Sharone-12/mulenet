import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { C, num } from '../theme';

function useSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

// Radii in screen pixels. The default nodeVal/nodeRelSize path renders area,
// so a "9" became a 3px dot and the whole graph read as scatter. Drawing the
// marks directly is the only way to guarantee a legible size.
const R = { controller: 7.5, mule: 5, clean: 2.2 };

export default function GraphPanel({ data, revealed, focusRing, onPickRing }) {
  const boxRef = useRef(null);
  const fgRef = useRef(null);
  const { width, height } = useSize(boxRef);
  const [hovered, setHovered] = useState(null);

  // react-force-graph mutates what it is given, so it always gets copies.
  const graph = useMemo(
    () => ({
      nodes: (data?.nodes ?? []).map((n) => ({ ...n })),
      links: (data?.links ?? []).map((l) => ({ ...l })),
    }),
    [data]
  );

  // One label per ring, anchored to its busiest account, so the fan-outs are
  // identifiable rather than anonymous starbursts.
  const labelAnchors = useMemo(() => {
    const degree = new Map();
    for (const l of graph.links) {
      for (const end of [l.source, l.target]) {
        const id = typeof end === 'object' ? end.id : end;
        degree.set(id, (degree.get(id) ?? 0) + 1);
      }
    }
    const best = new Map();
    for (const n of graph.nodes) {
      if (!n.ring_id) continue;
      const d = degree.get(n.id) ?? 0;
      const cur = best.get(n.ring_id);
      if (!cur || d > cur.d) best.set(n.ring_id, { id: n.id, d });
    }
    return new Set([...best.values()].map((v) => v.id));
  }, [graph]);

  const paint = useCallback(
    (node, ctx, globalScale) => {
      const role = revealed ? node.role : 'clean';
      const dimmed = focusRing && node.ring_id !== focusRing;
      const r = R[role] / globalScale;

      ctx.globalAlpha = dimmed ? 0.18 : 1;
      ctx.fillStyle =
        role === 'controller' ? C.markController : role === 'mule' ? C.markMule : C.markClean;
      ctx.strokeStyle = C.paper;
      ctx.lineWidth = 1.4 / globalScale;

      if (role === 'controller') {
        // Shape carries role too: darkened red and amber are near-identical
        // under deuteranopia, so colour alone cannot separate them.
        ctx.beginPath();
        ctx.rect(node.x - r, node.y - r, r * 2, r * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
        ctx.fill();
        if (role !== 'clean') ctx.stroke();
      }

      const showLabel =
        revealed && !dimmed && globalScale > 0.55 && labelAnchors.has(node.id) && node.ring_id;
      if (showLabel) {
        const size = 11 / globalScale;
        ctx.font = `800 ${size}px Archivo, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        const text = `Ring ${node.ring_id}`;
        const pad = 3 / globalScale;
        const w = ctx.measureText(text).width;
        const y = node.y - r - 3 / globalScale;
        ctx.globalAlpha = dimmed ? 0.18 : 0.92;
        ctx.fillStyle = C.ink;
        ctx.beginPath();
        ctx.roundRect(node.x - w / 2 - pad, y - size - pad, w + pad * 2, size + pad * 2, 3 / globalScale);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.fillText(text, node.x, y);
      }

      ctx.globalAlpha = 1;
    },
    [revealed, focusRing, labelAnchors]
  );

  // Keeps click/hover targets aligned with the drawn marks.
  const paintHitArea = useCallback(
    (node, color, ctx, globalScale) => {
      const r = (R[revealed ? node.role : 'clean'] + 3) / globalScale;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fill();
    },
    [revealed]
  );

  useEffect(() => {
    if (!focusRing || !fgRef.current) return undefined;
    const t = setTimeout(() => {
      fgRef.current?.zoomToFit(800, 90, (n) => n.ring_id === focusRing);
    }, 400);
    return () => clearTimeout(t);
  }, [focusRing, graph]);

  // Push disconnected rings apart so they read as separate structures rather
  // than drifting into one another.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !graph.nodes.length) return;
    fg.d3Force('charge')?.strength(-38).distanceMax(190);
    fg.d3Force('link')?.distance(22);
    fg.d3ReheatSimulation?.();
  }, [graph]);

  const flagged = graph.nodes.filter((n) => n.role !== 'clean').length;

  return (
    <div className="graphwrap" ref={boxRef}>
      <div className="graphbadge">
        {num(graph.nodes.length)} accounts · {num(graph.links.length)} transfers
        {data?.hops ? ` · ${data.hops}-hop` : ''}
        {revealed ? ` · ${flagged} flagged` : ''}
      </div>

      {hovered && (
        <div className="graphhover">
          <strong>{hovered.id}</strong>
          <span>
            {hovered.role === 'clean' ? 'not flagged' : hovered.role}
            {hovered.ring_id ? ` · ring ${hovered.ring_id}` : ''}
          </span>
        </div>
      )}

      {revealed && (
        <div className="legend">
          <span><i className="sq" style={{ background: C.markController }} /> controller</span>
          <span><i style={{ background: C.markMule }} /> mule</span>
          <span><i style={{ background: C.markClean }} /> not flagged</span>
        </div>
      )}

      {width > 0 && height > 0 && (
        <ForceGraph2D
          ref={fgRef}
          width={width}
          height={height}
          graphData={graph}
          backgroundColor={C.paper}
          nodeCanvasObject={paint}
          nodePointerAreaPaint={paintHitArea}
          linkColor={(l) =>
            focusRing && l.source?.ring_id !== focusRing && l.target?.ring_id !== focusRing
              ? 'rgba(10,10,10,0.07)'
              : 'rgba(10,10,10,0.3)'
          }
          linkWidth={1}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          onNodeHover={setHovered}
          onNodeClick={(n) => n.ring_id && onPickRing?.(n.ring_id)}
          cooldownTicks={120}
          warmupTicks={40}
          enableNodeDrag={false}
        />
      )}
    </div>
  );
}
