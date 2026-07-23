import { num } from '../theme';

const pct = (v, dp = 2) => `${(v * 100).toFixed(dp)}%`;

export default function Benchmark({ data }) {
  if (!data) return <div className="empty">Loading the comparison…</div>;
  if (data.error) return <div className="empty">{data.error}</div>;

  // Reading straight into a nested key threw and rendered a blank panel when
  // the exported bundle omitted the block. Name the problem instead.
  const b = data.benchmark;
  const c = data.comparison;
  if (!b || !c) {
    return (
      <div className="empty">
        Comparison data is missing from this build.
        <br />
        Regenerate it with <code>python scripts/export_bundle.py</code>.
      </div>
    );
  }

  const r = b.rule_engine;
  const m = b.mulenet;
  const top = b.mulenet_by_depth?.[0];

  return (
    <div className="bench">
      <div className="callout">
        Same {num(b.transactions_scanned)} transactions, same{' '}
        {num(b.laundering_total)} labelled laundering rows. MuleNet raises{' '}
        <b className="good">{Math.round(r.flagged / m.flagged)}× fewer alerts</b> and still catches{' '}
        <b className="good">{num(m.caught - r.caught)} more</b> of them.
      </div>

      <div className="verdict">
        <div className="vbox bad">
          <span className="t">Threshold Rules</span>
          <span className="big">{num(r.flagged)}</span>
          <span className="d">
            transactions flagged — {pct(r.flag_rate)} of everything.
            <br />
            Caught {num(r.caught)} of {num(b.laundering_total)} · recall {pct(r.recall)}
            <br />
            Precision {pct(r.precision, 4)} · ~{num(r.alerts_per_case)} alerts per real case.
          </span>
        </div>
        <div className="vbox good">
          <span className="t">MuleNet Graph Engine</span>
          <span className="big">{num(m.flagged)}</span>
          <span className="d">
            transactions flagged — {pct(m.flag_rate)} of everything.
            <br />
            Caught {num(m.caught)} of {num(b.laundering_total)} · recall {pct(m.recall)}
            <br />
            Precision {pct(m.precision, 4)} · ~{num(m.alerts_per_case)} alerts per real case.
          </span>
        </div>
      </div>

      <div className="scroller">
        <table>
          <thead>
            <tr>
              <th>Measured against IBM's labels</th>
              <th className="num">Threshold rules</th>
              <th className="num">MuleNet</th>
              <th className="num">Better by</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Transactions flagged</td>
              <td className="num">{num(r.flagged)}</td>
              <td className="num">{num(m.flagged)}</td>
              <td className="num"><span className="win">{Math.round(r.flagged / m.flagged)}× fewer</span></td>
            </tr>
            <tr>
              <td>Real laundering caught</td>
              <td className="num">{num(r.caught)}</td>
              <td className="num">{num(m.caught)}</td>
              <td className="num"><span className="win">+{num(m.caught - r.caught)}</span></td>
            </tr>
            <tr>
              <td>Laundering missed</td>
              <td className="num">{num(r.missed)}</td>
              <td className="num">{num(m.missed)}</td>
              <td className="num"><span className="win">{num(r.missed - m.missed)} fewer</span></td>
            </tr>
            <tr>
              <td>Recall</td>
              <td className="num">{pct(r.recall)}</td>
              <td className="num">{pct(m.recall)}</td>
              <td className="num"><span className="win">+{((m.recall - r.recall) * 100).toFixed(1)} pts</span></td>
            </tr>
            <tr>
              <td>Precision</td>
              <td className="num">{pct(r.precision, 4)}</td>
              <td className="num">{pct(m.precision, 4)}</td>
              <td className="num"><span className="win">{(m.precision / r.precision).toFixed(1)}×</span></td>
            </tr>
            <tr>
              <td>Alerts worked per real case</td>
              <td className="num">{num(r.alerts_per_case)}</td>
              <td className="num">{num(m.alerts_per_case)}</td>
              <td className="num"><span className="win">{Math.round(r.alerts_per_case / m.alerts_per_case)}× less work</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      {top && (
        <div className="callout">
          At the depth an analyst actually works — the top of the queue —{' '}
          <b className="good">{pct(top.precision)} of flagged transactions are real laundering</b>,
          against a base rate of {pct(b.base_rate, 4)}. That is a{' '}
          <b className="good">{num(b.lift_at_top)}× lift</b> over chance.
        </div>
      )}

      <div className="scroller">
        <table>
          <thead>
            <tr>
              <th>MuleNet queue depth</th>
              <th className="num">Flagged</th>
              <th className="num">Caught</th>
              <th className="num">Precision</th>
              <th className="num">Recall</th>
            </tr>
          </thead>
          <tbody>
            {b.mulenet_by_depth.map((d) => (
              <tr key={d.depth}>
                <td>Top {num(d.depth)} rings</td>
                <td className="num">{num(d.flagged)}</td>
                <td className="num">{num(d.caught)}</td>
                <td className="num">{pct(d.precision)}</td>
                <td className="num">{pct(d.recall)}</td>
              </tr>
            ))}
            <tr>
              <td>Full queue ({num(b.rings_in_queue)} rings)</td>
              <td className="num">{num(m.flagged)}</td>
              <td className="num">{num(m.caught)}</td>
              <td className="num">{pct(m.precision)}</td>
              <td className="num">{pct(m.recall)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="scroller">
        <table>
          <thead>
            <tr>
              <th>Threshold rule</th>
              <th className="num">Flags</th>
              <th className="num">Laundering caught</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.rules).map(([name, rule]) => (
              <tr key={name}>
                <td>{rule.description}</td>
                <td className="num">{num(rule.flags)}</td>
                <td className="num">{num(rule.laundering_caught)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="note">
        Both engines are scored identically: against the {num(b.laundering_total)} transactions IBM
        labels as laundering. MuleNet detects rings rather than individual payments, so a transaction
        counts as flagged when both of its accounts sit inside a detected ring — otherwise "2 million
        flags" versus "25 rings" would not be a comparison at all. Precision is low for both because
        the base rate is {pct(b.base_rate, 4)}; that is why lift, not raw precision, is the honest
        measure on a problem this rare.
      </p>
    </div>
  );
}
