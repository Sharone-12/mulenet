import { useCallback, useEffect, useRef, useState } from 'react';
import Benchmark from './components/Benchmark';
import ChatPanel from './components/ChatPanel';
import GraphPanel from './components/GraphPanel';
import Rail from './components/Rail';
import RingInspector from './components/RingInspector';
import RingQueue from './components/RingQueue';
import StatCards from './components/StatCards';
import { loadBundle, streamChat } from './api';
import { VIEWS } from './theme';

const GREETING =
  'I read a transaction graph of 6.9M transfers between 706K accounts. Ask me to scan it, ' +
  'investigate a ring, or write an investigation report.';

const HEADINGS = {
  network: ['Network', 'live graph'],
  rings: ['Ring Queue', 'ranked alerts'],
  inspect: ['Investigation', 'evidence'],
  benchmark: ['Rule Engine vs MuleNet', 'benchmark'],
};

export default function App() {
  const [view, setView] = useState('network');
  const [messages, setMessages] = useState([{ role: 'agent', text: GREETING }]);

  const [busy, setBusy] = useState(false);

  const [graph, setGraph] = useState(null);
  const [rings, setRings] = useState([]);
  const [ruleData, setRuleData] = useState(null);

  const [revealed, setRevealed] = useState(false);
  const [focusRing, setFocusRing] = useState(null);
  const [activeRing, setActiveRing] = useState(null);
  const [stats, setStats] = useState({ accounts: 0, rings: 0, amount: 0, currency: '' });

  const bundleRef = useRef(null);
  const messagesRef = useRef(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    loadBundle()
      .then((b) => {
        bundleRef.current = b;
        setGraph(b.graph);
        setRings(b.rings);
        setRuleData(b.rule_engine);
      })
      .catch(() => {});
  }, []);

  // Visuals are driven by tool results, never by parsing the narration, so
  // the graph and the agent can never disagree.
  const applyToolResult = useCallback((tool, result) => {
    if (!result || result.error) return;

    if (tool === 'scan_network') {
      const byCurrency = result.total_traced_by_currency ?? {};
      const [currency, amount] = Object.entries(byCurrency).sort((a, b) => b[1] - a[1])[0] ?? ['', 0];
      setStats({
        accounts: result.total_accounts_flagged ?? 0,
        rings: result.rings_detected ?? 0,
        amount: Math.round(amount),
        currency,
      });
      setRevealed(true);
      setFocusRing(null);
    }

    if (tool === 'investigate_ring' && result.ring_id) {
      setRevealed(true);
      setFocusRing(result.ring_id);
      setActiveRing(result.ring_id);
    }
  }, []);

  const send = useCallback(
    async (text) => {
      setMessages((p) => [...p, { role: 'user', text }]);
      setBusy(true);

      // Only narration is replayed as history; tool payloads are re-fetched
      // server-side, keeping each request small.
      const history = messagesRef.current
        .filter((m) => m.role === 'user' || m.role === 'agent')
        .slice(-6)
        .map((m) => ({ role: m.role === 'agent' ? 'assistant' : 'user', content: m.text }));

      try {
        for await (const e of streamChat(text, history)) {
          if (e.type === 'tool_call') {
            setMessages((p) => [...p, { role: 'tool', tool: e.tool, arguments: e.arguments }]);
          } else if (e.type === 'tool_result') {
            applyToolResult(e.tool, e.result);
          } else if (e.type === 'token') {
            setMessages((p) => {
              const last = p[p.length - 1];
              if (last?.role === 'agent' && last.streaming) {
                const copy = [...p];
                copy[copy.length - 1] = { ...last, text: last.text + e.text };
                return copy;
              }
              return [...p, { role: 'agent', text: e.text, streaming: true }];
            });
          } else if (e.type === 'error') {
            setMessages((p) => [...p, { role: 'agent', text: `Error: ${e.message}` }]);
          }
        }
      } catch (err) {
        setMessages((p) => [...p, { role: 'agent', text: `Error: ${err.message}` }]);
      } finally {
        setBusy(false);
        setMessages((p) => p.map((m, i) => (i === p.length - 1 ? { ...m, streaming: false } : m)));
      }
    },
    [applyToolResult]
  );

  const openRing = useCallback(
    (ringId) => {
      setActiveRing(ringId);
      setFocusRing(ringId);
      setRevealed(true);
      setView('inspect');
    },
    []
  );

  const generateSar = useCallback(
    (ringId) => send(`Write the investigation report for ring ${ringId}`),
    [send]
  );

  const [title, sub] = HEADINGS[view] ?? HEADINGS.network;

  return (
    <div className="app">
      <Rail
        view={view}
        setView={setView}
        onScan={() => send('Scan the network for suspicious activity')}
        connected
        busy={busy}
      />

      <ChatPanel messages={messages} connected busy={busy} onSend={send} />

      <section className="panel work">
        <div className="panel-head">
          <span
            style={{
              width: 11,
              height: 11,
              borderRadius: 3,
              background: VIEWS.find((v) => v.id === view)?.color,
            }}
          />
          <h2>{title}</h2>
          <span className="sub">{sub}</span>
        </div>

        {view === 'network' && (
          <>
            <StatCards
              accounts={stats.accounts}
              rings={stats.rings}
              amount={stats.amount}
              currency={stats.currency}
            />
            <GraphPanel
              data={graph}
              revealed={revealed}
              focusRing={focusRing}
              onPickRing={openRing}
            />
          </>
        )}

        {view === 'rings' && (
          <div className="scroll">
            <RingQueue rings={rings} onSelect={openRing} />
          </div>
        )}

        {view === 'inspect' && (
          <div className="scroll">
            <RingInspector ringId={activeRing} onGenerateSar={generateSar} />
          </div>
        )}

        {view === 'benchmark' && (
          <div className="scroll">
            <Benchmark data={ruleData} />
          </div>
        )}
      </section>
    </div>
  );
}
