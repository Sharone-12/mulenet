import { useEffect, useRef, useState } from 'react';
import Markdown from './Markdown';

// What the agent actually did, phrased as a completed step rather than a
// status - by the time the chip is on screen the call has already returned.
const TOOL_LABELS = {
  scan_network: 'Scanned the network',
  investigate_ring: 'Pulled ring',
  classify_roles: 'Classified roles in ring',
};

const SUGGESTIONS = [
  'Scan the network',
  'Tell me about ring 1',
  "Who's running it?",
  'Write the investigation report',
];

// A SAR arrives as ordinary narration, so it is recognised by shape. The
// model is not reliable enough to be asked to flag it.
const isSar = (text) => /suspicious activity report/i.test(text) && text.length > 400;

export default function ChatPanel({ messages, connected, busy, onSend }) {
  const [draft, setDraft] = useState('');
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  const submit = (e) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy || !connected) return;
    onSend(text);
    setDraft('');
  };

  return (
    <section className="panel chat">
      <div className="panel-head">
        <h2>Investigator</h2>
        <span className="sub">agent</span>
      </div>

      <div className="scroll">
        <div className="msgs">
          {messages.map((m, i) => {
            if (m.role === 'tool') {
              return (
                <div className="chip" key={i}>
                  <span className="chip-dot" aria-hidden />
                  {TOOL_LABELS[m.tool] || m.tool}
                  {m.arguments?.ring_id != null ? ` ${m.arguments.ring_id}` : ''}
                </div>
              );
            }
            if (m.role === 'agent' && isSar(m.text)) {
              return (
                <div className="sar" key={i}>
                  <div className="sar-head">
                    <span aria-hidden>▣</span> Suspicious Activity Report
                  </div>
                  <div className="sar-body">
                    <Markdown text={m.text} />
                  </div>
                </div>
              );
            }
            return (
              <div className={`bubble ${m.role}`} key={i}>
                {m.role === 'agent' ? <Markdown text={m.text} /> : m.text}
              </div>
            );
          })}
          <div ref={endRef} />
        </div>
      </div>

      {messages.length <= 1 && (
        <div className="suggest">
          {SUGGESTIONS.map((s) => (
            <button type="button" key={s} onClick={() => onSend(s)} disabled={!connected || busy}>
              {s}
            </button>
          ))}
        </div>
      )}

      <form className="composer" onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={connected ? 'Ask about the network…' : 'Connecting to agent…'}
          disabled={!connected}
        />
        <button type="submit" disabled={!connected || busy || !draft.trim()}>
          {busy ? '···' : 'Send'}
        </button>
      </form>
    </section>
  );
}
