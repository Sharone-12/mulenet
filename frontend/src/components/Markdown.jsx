/**
 * Minimal markdown renderer for agent output.
 *
 * Builds React elements rather than an HTML string: model output is
 * untrusted, and in an AML tool it also carries attacker-influenced data
 * (account names, transaction memos), so it must never reach
 * dangerouslySetInnerHTML. Anything not handled here renders as plain text,
 * which is the safe failure.
 *
 * Supports the subset the agent actually emits: bold, italic, inline code,
 * bullet and numbered lists, and headings.
 */

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g;

function renderInline(text, keyPrefix) {
  return text
    .split(INLINE)
    .filter(Boolean)
    .map((part, i) => {
      const key = `${keyPrefix}-${i}`;
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
        return <strong key={key}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
        return <code key={key}>{part.slice(1, -1)}</code>;
      }
      if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        return <em key={key}>{part.slice(1, -1)}</em>;
      }
      return <span key={key}>{part}</span>;
    });
}

const BULLET = /^\s*[*-]\s+(.*)$/;
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;
const HEADING = /^\s*(#{1,4})\s+(.*)$/;

export default function Markdown({ text }) {
  if (!text) return null;

  const lines = text.split('\n');
  const blocks = [];
  let list = null;
  let para = [];

  const flushPara = () => {
    if (!para.length) return;
    const body = para.join(' ');
    blocks.push(
      <p key={`p${blocks.length}`} className="md-p">
        {renderInline(body, `p${blocks.length}`)}
      </p>
    );
    para = [];
  };

  const flushList = () => {
    if (!list) return;
    const Tag = list.ordered ? 'ol' : 'ul';
    blocks.push(
      <Tag key={`l${blocks.length}`} className="md-list">
        {list.items.map((item, i) => (
          <li key={i}>{renderInline(item, `l${blocks.length}-${i}`)}</li>
        ))}
      </Tag>
    );
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      flushPara();
      flushList();
      blocks.push(
        <div key={`h${blocks.length}`} className="md-h">
          {renderInline(heading[2], `h${blocks.length}`)}
        </div>
      );
      continue;
    }

    // A line that is entirely bold is a section label, which is how the SAR
    // narrative marks its sections ("**Summary:**").
    if (/^\*\*[^*]+:?\*\*:?$/.test(line.trim())) {
      flushPara();
      flushList();
      blocks.push(
        <div key={`h${blocks.length}`} className="md-h">
          {line.trim().replace(/^\*\*|\*\*:?$/g, '')}
        </div>
      );
      continue;
    }

    const numbered = line.match(NUMBERED);
    if (numbered) {
      flushPara();
      if (!list?.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(numbered[2]);
      continue;
    }

    const bullet = line.match(BULLET);
    if (bullet) {
      flushPara();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1]);
      continue;
    }

    flushList();
    para.push(line.trim());
  }

  flushPara();
  flushList();

  return <div className="md">{blocks}</div>;
}
