"""Deterministic submission flags that don't need a model.

grader_directed(): text that addresses the scorer instead of the topic ("ignore previous
instructions", "rate this 100"). It feeds flags.gaming together with the LLM judge's own
gaming verdict when one is available.
"""

import re

_PATTERNS = [
    # "ignore / disregard ... previous / prior / all ... instructions / prompt / rules"
    re.compile(r"\b(ignore|disregard|forget)\b[^.!?]{0,40}\b(previous|prior|above|earlier|all|your)\b"
               r"[^.!?]{0,20}\b(instructions?|prompts?|rules|guidelines)\b", re.I),
    # "rate / score / grade ... this / me ... 100 / 10/10 / maximum / perfect"
    re.compile(r"\b(rate|score|grade|mark|give)\b[^.!?]{0,30}\b(this|me|it|submission|answer)\b"
               r"[^.!?]{0,30}(\b100\b|\b10\s*/\s*10\b|\bmaximum\b|\bmax\b|\bperfect\b|\bfull marks\b|\bhighest\b)", re.I),
    # direct address of the evaluator: "dear grader", "note to the judge", "as the AI evaluator you must"
    re.compile(r"\b(dear|note to|attention|hey)\b[^.!?]{0,10}\b(grader|judge|evaluator|scorer|reviewer ai|ai)\b", re.I),
    re.compile(r"\b(set|output|return)\b[^.!?]{0,20}\b(overall_novelty|novelty score|system_score|all flags)\b", re.I),
]


def grader_directed(*texts: str) -> bool:
    return any(p.search(t or "") for t in texts for p in _PATTERNS)
