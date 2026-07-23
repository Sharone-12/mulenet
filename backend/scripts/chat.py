"""Talk to the MuleNet agent in the terminal.

    python scripts/chat.py                    # interactive
    python scripts/chat.py --demo             # run the scripted demo questions
    python scripts/chat.py -m "your question"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mulenet import agent as agent_mod, config  # noqa: E402

DEMO_QUESTIONS = [
    "Scan the network for suspicious activity",
    "Tell me about ring 1",
    "Who's running that ring?",
    "Write the investigation report for ring 1",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", help="ask one question and exit")
    ap.add_argument("--demo", action="store_true", help="run the scripted demo")
    args = ap.parse_args()

    try:
        agent = agent_mod.Agent()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    except ImportError:
        print("FAIL: groq package not installed. Run: pip install groq")
        return 1

    print(f"MuleNet agent - model {config.GROQ_MODEL}\n")

    if args.demo:
        for question in DEMO_QUESTIONS:
            print(f"> {question}")
            print(agent.ask(question), "\n")
            print(f"  [tools called: {[t['tool'] for t in agent.tool_log]}]\n")
        return 0

    if args.message:
        print(agent.ask(args.message))
        return 0

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question in {"exit", "quit"}:
            return 0
        if question:
            print(agent.ask(question), "\n")


if __name__ == "__main__":
    raise SystemExit(main())
