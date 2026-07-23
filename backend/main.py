"""MuleNet HTTP + websocket API.

A transport layer over the tested Phase 1 engine. No detection logic lives
here: every endpoint delegates to mulenet.tools, mulenet.viz or
mulenet.rule_engine.

    uvicorn main:app --port 8000
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mulenet import agent as agent_mod, config, engine, keys, rule_engine, tools, viz

app = FastAPI(title="MuleNet API", version="1.0")

# Open CORS: this is a local demo served from the Vite dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    top_k: int = 25


class RingRequest(BaseModel):
    ring_id: int


@app.on_event("startup")
def warm_caches() -> None:
    """Load the engine and rule-engine results before serving.

    Both take ~13s cold. Paying that during a demo's first question would
    look like a hang, so it is paid at boot instead.
    """
    engine.load()
    rule_engine.evaluate()
    viz.graph_sample()


@app.get("/health")
def health() -> dict:
    eng = engine.load()
    return {
        "status": "ok",
        "model": config.GROQ_MODEL,
        "rings_in_queue": int(len(eng.rings)),
        "transactions": int(len(eng.transactions)),
        "keys": keys.pool().status(),
    }


@app.post("/tools/scan_network")
def scan_network(req: ScanRequest) -> dict:
    return tools.scan_network(top_k=req.top_k)


@app.post("/tools/investigate_ring")
def investigate_ring(req: RingRequest) -> dict:
    return tools.investigate_ring(req.ring_id)


@app.post("/tools/classify_roles")
def classify_roles(req: RingRequest) -> dict:
    return tools.classify_roles(req.ring_id)


@app.get("/tools/rule_engine")
def rule_engine_endpoint() -> dict:
    """Threshold-rule baseline, alongside the graph engine's numbers."""
    result = dict(rule_engine.evaluate())
    eng = engine.load()
    scan = tools.scan_network(top_k=25)
    result["comparison"] = {
        "rule_engine_flags": result["total_flags"],
        "rule_engine_caught": result["laundering_caught"],
        "rule_engine_missed": result["laundering_missed"],
        "graph_engine_rings": int(len(eng.rings)),
        "graph_engine_top_rings": scan["rings_detected"],
        "graph_engine_accounts_traced": scan["total_accounts_flagged"],
    }
    return result


@app.get("/graph/sample")
def graph_sample(top_k: int = 25, hops: int = 1) -> dict:
    return viz.graph_sample(top_k=top_k, hops=hops)


@app.get("/graph/ring/{ring_id}")
def graph_ring(ring_id: int) -> dict:
    return viz.ring_subgraph(ring_id)


@app.get("/rings")
def ring_list(top_k: int = 25) -> list[dict]:
    return viz.ring_cards(top_k=top_k)


SAR_PROMPT = """Write a formal Suspicious Activity Report narrative for the ring below.

Requirements:
- Formal regulatory register, written for a bank compliance officer.
- Use ONLY the figures given. Never invent an account, amount, date or place.
- State amounts with the currency exactly as given. Do not convert currencies.
- Structure: Summary, Account Activity, Flow of Funds, Role Assessment,
  Recommended Action.
- Note explicitly that role classification is heuristic and requires
  analyst confirmation.

RING DATA
{ring}

ROLE CLASSIFICATION
{roles}
"""


@app.post("/tools/generate_sar")
def generate_sar(req: RingRequest) -> dict:
    """LLM-written SAR narrative grounded in one ring's data."""
    ring = tools.investigate_ring(req.ring_id)
    if "error" in ring:
        return ring
    roles = tools.classify_roles(req.ring_id)

    prompt = SAR_PROMPT.format(
        ring=json.dumps(ring, indent=2, default=str),
        roles=json.dumps(roles, indent=2, default=str),
    )

    try:
        # Same pool the agent uses, so SAR generation fails over too and
        # shares the exhaustion state rather than re-burning a dead key.
        response = keys.pool().complete(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": agent_mod.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "ring_id": req.ring_id,
        "report": response.choices[0].message.content,
        "ring": ring,
        "roles": roles,
    }


async def _pump(session: "agent_mod.Agent", message: str):
    """Drive the synchronous agent loop without blocking the event loop.

    `ask_events` makes blocking HTTP calls to Groq. Iterating it directly in
    the websocket handler starves the event loop, so the keepalive ping never
    goes out and the connection is dropped mid-conversation with a 1011. The
    generator therefore runs on a worker thread and hands events back through
    a queue.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def produce() -> None:
        try:
            for event in session.ask_events(message):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    task = loop.run_in_executor(None, produce)
    try:
        while True:
            event = await queue.get()
            if event is sentinel:
                return
            yield event
    finally:
        await task


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    """Agent loop over a socket.

    Tool results are forwarded as their own messages so the frontend can drive
    the graph and stat cards from structured data rather than scraping the
    narration.
    """
    await websocket.accept()
    try:
        session = agent_mod.Agent()
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    try:
        while True:
            payload = await websocket.receive_text()
            try:
                message = json.loads(payload).get("message", "")
            except json.JSONDecodeError:
                message = payload

            if not message.strip():
                continue

            async for event in _pump(session, message):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        return
