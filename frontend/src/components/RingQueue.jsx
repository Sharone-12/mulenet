import RingTopology from './RingTopology';
import { C, formatAmounts } from '../theme';

const SIGNAL_COLOR = {
  pass_through: C.orange,
  fan_out: C.blue,
  fan_in: C.blue,
  cycle: C.purple,
  velocity_burst: C.amber,
};

export default function RingQueue({ rings, onSelect }) {
  if (!rings.length) {
    return <div className="empty">Loading the alert queue…</div>;
  }

  return (
    <div className="ringgrid">
      {rings.map((r, i) => (
        <button type="button" className="ringcard" key={r.ring_id} onClick={() => onSelect(r.ring_id)}>
          <div className="top">
            <span className="rid">Ring {r.ring_id}</span>
            <span className="rank">#{i + 1}</span>
          </div>

          {/* The shape is the finding - a fan, a chain and a cycle are
              distinguishable here without opening the ring. */}
          <div className="thumb">
            <RingTopology ring={r.topology} roles={r.roles} compact />
          </div>

          <span className="amt">{formatAmounts(r.amount_by_currency, { compact: true })}</span>
          <span className="meta">
            {r.accounts} accounts · {r.transactions} transactions
          </span>
          <div className="tags">
            {(r.pattern ? r.pattern.split(', ') : []).map((p) => (
              <span className="tag" key={p} style={{ background: SIGNAL_COLOR[p] || C.paper }}>
                {p.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </button>
      ))}
    </div>
  );
}
