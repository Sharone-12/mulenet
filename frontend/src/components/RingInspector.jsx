import { useEffect, useState } from 'react';
import RingTimeline from './RingTimeline';
import RingTopology from './RingTopology';
import { API, formatAmounts } from '../theme';

export default function RingInspector({ ringId, onGenerateSar }) {
  const [ring, setRing] = useState(null);
  const [roles, setRoles] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ringId) return undefined;
    let live = true;
    setRing(null);
    setRoles(null);
    setError(null);

    const post = (path) =>
      fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ring_id: ringId }),
      }).then((r) => r.json());

    Promise.all([post('/tools/investigate_ring'), post('/tools/classify_roles')])
      .then(([r, c]) => {
        if (!live) return;
        if (r.error) setError(r.error);
        setRing(r);
        setRoles(c);
      })
      .catch((e) => live && setError(String(e)));

    return () => {
      live = false;
    };
  }, [ringId]);

  if (!ringId) {
    return <div className="empty">Pick a ring from the queue, or ask the agent to investigate one.</div>;
  }
  if (error) return <div className="empty">{error}</div>;
  if (!ring) return <div className="empty">Loading ring {ringId}…</div>;

  const roleOf = (account) => {
    if (roles?.controllers?.some((c) => c.account === account)) return 'controller';
    if (roles?.mules?.some((m) => m.account === account)) return 'mule';
    return '';
  };

  return (
    <div className="inspect">
      <div className="hero">
        <h3>Ring {ring.ring_id}</h3>
        <p>
          {ring.account_count} accounts moved {formatAmounts(ring.amount_by_currency)} across{' '}
          {ring.transaction_count} transactions over {ring.duration_hours} hours, between{' '}
          {ring.first_seen} and {ring.last_seen}.
        </p>
      </div>

      <div className="flow">
        {ring.source_accounts?.map((a) => (
          <span className={`acct ${roleOf(a)}`} key={a}>{a}</span>
        ))}
        {ring.source_accounts?.length > 0 && <span aria-hidden>→</span>}
        {ring.accounts
          .filter((a) => !ring.source_accounts?.includes(a) && !ring.cashout_accounts?.includes(a))
          .map((a) => (
            <span className={`acct ${roleOf(a)}`} key={a}>{a}</span>
          ))}
        {ring.cashout_accounts?.length > 0 && <span aria-hidden>→</span>}
        {ring.cashout_accounts?.map((a) => (
          <span className={`acct ${roleOf(a)}`} key={a}>{a}</span>
        ))}
      </div>

      <div className="viz">
        <div className="vizbox">
          <div className="vizhead">Structure</div>
          <RingTopology ring={ring} roles={roles} />
        </div>
        <div className="vizbox">
          <RingTimeline ring={ring} />
        </div>
      </div>

      <div className="kv">
        <div className="kvi">
          <div className="k">Duration</div>
          <div className="v">{ring.duration_hours}h</div>
        </div>
        <div className="kvi">
          <div className="k">Transactions</div>
          <div className="v">{ring.transaction_count}</div>
        </div>
        <div className="kvi">
          <div className="k">Signals</div>
          <div className="v" style={{ fontSize: 13 }}>{ring.signals?.join(', ')}</div>
        </div>
        <div className="kvi">
          <div className="k">Currencies</div>
          <div className="v" style={{ fontSize: 13 }}>{ring.currencies?.join(', ')}</div>
        </div>
      </div>

      {roles && !roles.error && (
        <>
          <div className="roles">
            <div className="rolebox ctrl">
              <div className="rh">Controllers · freeze &amp; investigate</div>
              <ul>
                {roles.controllers.length === 0 && <li><em>none identified</em></li>}
                {roles.controllers.map((c) => (
                  <li key={c.account}>
                    <code>{c.account}</code>
                    <em>{c.position} · {c.confidence}</em>
                  </li>
                ))}
              </ul>
            </div>
            <div className="rolebox mule">
              <div className="rh">Mules · contact &amp; warn</div>
              <ul>
                {roles.mules.length === 0 && <li><em>none identified</em></li>}
                {roles.mules.map((m) => (
                  <li key={m.account}>
                    <code>{m.account}</code>
                    <em>{m.hold_hours != null ? `held ${m.hold_hours}h` : m.position}</em>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <p className="note">
            Role assignment is heuristic scoring, not a validated classifier — IBM's data carries no
            role labels, so these require analyst confirmation.
          </p>
        </>
      )}

      <table className="txns">
        <thead>
          <tr>
            <th>From</th>
            <th>To</th>
            <th>When</th>
            <th style={{ textAlign: 'right' }}>Amount</th>
          </tr>
        </thead>
        <tbody>
          {ring.transactions?.map((t, i) => (
            <tr key={i}>
              <td className="mono">{t.from}</td>
              <td className="mono">{t.to}</td>
              <td>{t.timestamp}</td>
              <td className="num">{t.amount.toLocaleString()} {t.currency}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {ring.transactions_truncated > 0 && (
        <p className="note">{ring.transactions_truncated} further transactions not shown.</p>
      )}

      <button type="button" className="railbtn dark" onClick={() => onGenerateSar(ring.ring_id)}>
        Generate Suspicious Activity Report ↗
      </button>
    </div>
  );
}
