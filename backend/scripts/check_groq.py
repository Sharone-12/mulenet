"""Verify the Groq key and model work before the agent depends on them.

    python scripts/check_groq.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import config  # noqa: E402


def main() -> int:
    try:
        key = config.groq_api_key()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    try:
        from groq import Groq
    except ImportError:
        print("FAIL: groq package not installed. Run: pip install groq")
        return 1

    client = Groq(api_key=key)
    print(f"model: {config.GROQ_MODEL}")

    # A trivial tool definition, so this also proves tool-calling works on the
    # chosen model - that is the part the agent actually depends on.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scan_network",
                "description": "Scan the transaction graph for laundering rings.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": "Scan the network for suspicious activity."}],
            tools=tools,
            tool_choice="auto",
            max_tokens=256,
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    calls = resp.choices[0].message.tool_calls
    if not calls:
        print("WARN: key works, but the model answered without calling the tool.")
        print(f"  reply: {resp.choices[0].message.content!r}")
        return 1

    print(f"OK: key valid, tool-calling works (model chose {calls[0].function.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
