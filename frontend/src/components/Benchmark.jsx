import { num } from '../theme';

export default function Benchmark({ data }) {
  if (!data) return <div className="empty">Loading the comparison…</div>;
  if (data.error) return <div className="empty">{data.error}</div>;

  const c = data.comparison;
  const perHit = Math.round(c.rule_engine_flags / Math.max(1, c.rule_engine_caught));

  return (
    <div className="bench">
      <div className="callout">
        Same data. The rule engine raises{' '}
        <b className="bad">{num(c.rule_engine_flags)} flags</b> and still misses{' '}
        <b className="bad">{num(c.rule_engine_missed)}</b> of{' '}
        {num(data.total_laundering_transactions)} laundering transactions. MuleNet returns a{' '}
        <b className="good">ranked queue</b> — {c.graph_engine_top_rings} rings,{' '}
        {c.graph_engine_accounts_traced} accounts, worked top-down.
      </div>

      <div className="verdict">
        <div className="vbox bad">
          <span className="t">Threshold Rules</span>
          <span className="big">{num(c.rule_engine_flags)}</span>
          <span className="d">
            flags across {num(data.transactions_scanned)} transactions.
            <br />
            Recall {(data.recall * 100).toFixed(1)}% · precision {(data.precision * 100).toFixed(3)}%.
            <br />
            An analyst works ~{num(perHit)} alerts per real case.
          </span>
        </div>
        <div className="vbox good">
          <span className="t">MuleNet Graph Engine</span>
          <span className="big">{c.graph_engine_top_rings}</span>
          <span className="d">
            ranked rings surfaced, {c.graph_engine_accounts_traced} accounts traced.
            <br />
            {num(c.graph_engine_rings)} structures in the full queue.
            <br />
            Top 10 alerts run 78.6% precise.
          </span>
        </div>
      </div>

      <table className="txns">
        <thead>
          <tr>
            <th>Threshold rule</th>
            <th style={{ textAlign: 'right' }}>Flags</th>
            <th style={{ textAlign: 'right' }}>Laundering caught</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.rules).map(([name, r]) => (
            <tr key={name}>
              <td>{r.description}</td>
              <td className="num">{num(r.flags)}</td>
              <td className="num">{num(r.laundering_caught)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="note">
        Every rule inspects one transaction, or one account-day, in isolation — none can see that a
        set of accounts forms a structure. Cross-border is approximated by a mismatch between sent
        and received currency, as the dataset has no country field. Recall here is measured against
        all 3,565 labelled laundering transactions, the fairest basis for a transaction-level engine.
      </p>
    </div>
  );
}
