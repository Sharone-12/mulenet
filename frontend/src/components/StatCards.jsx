import { useEffect, useRef, useState } from 'react';
import { C } from '../theme';

/** Counts from the previous value to `target` so cards animate on a scan. */
function useCountUp(target, ms = 850) {
  const [value, setValue] = useState(0);
  const fromRef = useRef(0);

  useEffect(() => {
    const from = fromRef.current;
    if (from === target) return undefined;
    let raf;
    const t0 = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / ms);
      setValue(Math.round(from + (target - from) * (1 - (1 - t) ** 3)));
      if (t < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);

  return value;
}

function Stat({ label, value, unit, color }) {
  const shown = useCountUp(value);
  return (
    <div className="stat" style={{ background: color }}>
      <span className="k">{label}</span>
      <span className="v">{shown.toLocaleString()}</span>
      <span className="u">{unit}</span>
    </div>
  );
}

export default function StatCards({ accounts, rings, amount, currency }) {
  return (
    <div className="stats">
      <Stat label="Accounts Flagged" value={accounts} unit="in top alerts" color={C.blue} />
      <Stat label="Rings Detected" value={rings} unit="ranked structures" color={C.amber} />
      <Stat
        label="Amount Traced"
        value={amount}
        unit={currency ? `${currency} (largest)` : 'awaiting scan'}
        color={C.green}
      />
    </div>
  );
}
