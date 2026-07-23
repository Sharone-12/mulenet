"""Scope and prompt-injection guard for the agent.

Two layers, because neither is sufficient alone:

1. A deterministic pre-filter here. Override attempts ("ignore all previous
   instructions", "you are now…", "print your system prompt") are refused
   before a single token reaches the model, so there is nothing to talk
   around. This cannot be defeated by rephrasing the *instruction* because it
   never gets executed.

2. Scope rules in the system prompt, for the far larger space of merely
   off-topic questions, which no pattern list can enumerate.

The pre-filter is deliberately narrow. Over-blocking a real investigator
question is a worse failure than letting a weird-but-harmless one through -
the model still has its own instructions as a backstop.
"""

from __future__ import annotations

import re

REFUSAL = (
    "I only work on money laundering and financial crime analysis for this "
    "transaction dataset. I can scan the network for laundering rings, "
    "investigate a specific ring, classify controllers versus recruited mules, "
    "or write an investigation report. What would you like to look at?"
)

# Attempts to replace, reveal, or escape the agent's instructions. Anchored on
# the action rather than on topic words, so paraphrases of the same manoeuvre
# still land.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|earlier|above|your)\b",
    r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|earlier|above|your)\b",
    r"\bforget\s+(all\s+|everything\s+|your\s+|the\s+)?(you|instructions|rules|prompt|above)",
    r"\b(new|updated|revised)\s+(instructions?|rules?|prompt)\s*[:.\-]",
    r"\byou\s+are\s+now\s+(a|an|no longer)\b",
    r"\bpretend\s+(to\s+be|you\s+are|that\s+you)\b",
    r"\bact\s+as\s+(a|an|if\s+you)\b",
    r"\broleplay\b",
    r"\b(system|initial|original)\s+(prompt|instructions?|message)\b",
    r"\b(reveal|show|print|repeat|output|display)\s+(me\s+)?(your|the)\s+(prompt|instructions?|rules)\b",
    r"\b(developer|debug|god|admin)\s+mode\b",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"\bwithout\s+(any\s+)?(restrictions?|limits?|filters?|guardrails?)\b",
    r"\bdo\s+anything\s+now\b",
    r"\boverride\s+(your|the|all)\b",
    r"\bstop\s+being\b",
    r"\byour\s+(real|true|actual)\s+(instructions?|purpose|prompt)\b",
]

_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# Unambiguously off-topic asks that keep showing up in demos. This is a
# convenience shortcut, NOT the topic boundary - the model's scope rules are
# what actually hold the line.
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|bake|chocolate|cake|pizza)\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|song|story|essay|joke|rap)\b",
    r"\b(python|javascript|java|c\+\+|html|css|sql)\s+(code|script|function|program)\b",
    r"\bhomework\b",
    r"\b(weather|football|movie|lyrics|horoscope)\b",
    r"\btranslate\s+(this|the following|into)\b",
]

_OFF_TOPIC = re.compile("|".join(OFF_TOPIC_PATTERNS), re.IGNORECASE)


def screen(message: str) -> str | None:
    """Return a refusal if the message should never reach the model.

    `None` means let it through.
    """
    if not message or not message.strip():
        return None

    if _INJECTION.search(message):
        return REFUSAL

    if _OFF_TOPIC.search(message):
        return REFUSAL

    return None
