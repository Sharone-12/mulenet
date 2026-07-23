import Logo from './Logo';
import { C, VIEWS } from '../theme';

export default function Rail({ view, setView, onScan, connected, busy }) {
  return (
    <nav className="rail">
      <div className="brand">
        <div className="brand-row">
          <Logo size={32} />
          <h1>mulenet.</h1>
        </div>
        <p>laundering ring detection</p>
      </div>

      {VIEWS.map((v) => (
        <button
          type="button"
          key={v.id}
          className={`navcard ${view === v.id ? 'active' : ''}`}
          style={{ background: v.color }}
          onClick={() => setView(v.id)}
          aria-current={view === v.id ? 'page' : undefined}
        >
          <span className="n">{v.n}</span>
          <span className="arrow" aria-hidden>↗</span>
          <span className="lbl">{v.label}</span>
        </button>
      ))}

      <button
        type="button"
        className="railbtn"
        style={{ background: C.purple }}
        onClick={onScan}
        disabled={!connected || busy}
      >
        {busy ? 'Scanning…' : 'Scan the Network'}
      </button>

      <div className="rail-status">
        <i className={`led ${connected ? 'on' : ''}`} />
        {connected ? 'Agent online' : 'Agent offline'}
      </div>
    </nav>
  );
}
