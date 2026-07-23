/**
 * Data access.
 *
 * Read-only findings come from a precomputed 100KB bundle rather than a live
 * API: detection runs offline, so nothing needs pandas, NetworkX or the 284MB
 * parquet cache at request time. That is what makes the app deployable to a
 * serverless host.
 *
 * Chat streams over SSE rather than a websocket, because Vercel Functions do
 * not run long-lived websocket servers. The event shapes are unchanged, so the
 * UI handles both identically.
 */

let bundlePromise = null;

export function loadBundle() {
  if (!bundlePromise) {
    bundlePromise = fetch('/data/mulenet.json').then((r) => {
      if (!r.ok) throw new Error(`bundle ${r.status}`);
      return r.json();
    });
  }
  return bundlePromise;
}

/**
 * POST a message and yield agent events as they stream in.
 *
 * `history` carries only prior narration; tool payloads are re-fetched
 * server-side when needed, which keeps each request small.
 */
export async function* streamChat(message, history, { signal } = {}) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal,
  });

  if (!res.ok || !res.body) {
    yield { type: 'error', message: `agent unavailable (${res.status})` };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a partial frame stays in the
    // buffer until the rest of it arrives.
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data:'));
      if (!line) continue;
      try {
        yield JSON.parse(line.slice(5).trim());
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}
