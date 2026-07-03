"""
security.py
-----------
Guards against prompt-injection payloads hidden inside uploaded documents.

Two layers of defence:

1. Static pattern scan  — fast regex check on raw extracted text before
   any embedding.  Flags text that looks like it is trying to override
   the system prompt, exfiltrate data, or impersonate the assistant.

2. LLM-assisted scan (optional, async-friendly) — send a small sample
   of the document to the local LLM and ask it to rate injection risk.
   Only used for documents that pass the static scan but are still
   suspicious (e.g. unusual Unicode, excessive instruction-like text).

Usage
-----
    from core.security import scan_chunks, SecurityViolation

    try:
        scan_chunks(chunks, source_filename="report.pdf")
    except SecurityViolation as exc:
        # reject the upload and return HTTP 422
        raise HTTPException(status_code=422, detail=str(exc))
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from langchain.schema import Document

# ---------------------------------------------------------------------------
# Injection pattern library
# ---------------------------------------------------------------------------

# Each tuple: (pattern, human-readable label)
_INJECTION_PATTERNS: List[tuple] = [
    # Direct system-prompt overrides
    (r"(?i)\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)\b",
     "instruction-override attempt"),
    (r"(?i)\byou\s+are\s+now\s+(a\s+)?(different|new|evil|unrestricted|jailbroken|DAN)\b",
     "persona-hijack attempt"),
    (r"(?i)\bact\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(evil|uncensored|unfiltered|jailbroken)\b",
     "persona-hijack attempt"),

    # Prompt delimiter injections (trying to close/open system blocks)
    (r"<\|?system\|?>|<\|?im_start\|?>|<\|?endoftext\|?>|\[INST\]|\[/INST\]",
     "prompt-delimiter injection"),
    (r"###\s*(System|Human|Assistant|Instruction)",
     "chat-template injection"),

    # Data exfiltration via tool calls or URLs
    (r"(?i)(send|post|fetch|retrieve|exfiltrate|leak)\s+.{0,60}(password|secret|token|api.?key|credential)",
     "credential-exfiltration pattern"),
    (r"(?i)curl\s+https?://|wget\s+https?://|requests\.get\(",
     "outbound HTTP call embedded in document"),

    # Classic jailbreaks
    (r"(?i)\bDAN\b.{0,40}(mode|activated|enabled)",
     "DAN jailbreak"),
    (r"(?i)do\s+anything\s+now",
     "DAN jailbreak"),
    (r"(?i)\bjailbreak\b",
     "jailbreak keyword"),

    # Invisible / control characters used to hide injections
    (r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]",
     "zero-width / direction-override characters"),
]

_COMPILED: List[tuple] = [
    (re.compile(pat), label) for pat, label in _INJECTION_PATTERNS
]

# Max fraction of chunks that may contain instruction-like text before
# we treat the *whole document* as suspicious.
_MAX_INSTRUCTION_DENSITY = 0.15   # 15 %

# Patterns that count as "instruction-like" but not immediately fatal
_INSTRUCTION_LIKE = re.compile(
    r"(?i)\b(you (must|should|will|shall)|always respond|never say|"
    r"your (task|goal|job|role|purpose) is|respond only in|"
    r"output (only|exactly|must be))\b"
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class SecurityViolation(ValueError):
    """Raised when a document fails the security scan."""


@dataclass
class ScanResult:
    passed: bool
    violations: List[str]
    instruction_density: float


def scan_text(text: str) -> ScanResult:
    """
    Scan a single block of text for injection patterns.

    Returns a ScanResult; does NOT raise — callers decide what to do.
    """
    violations: List[str] = []
    for pattern, label in _COMPILED:
        if pattern.search(text):
            violations.append(label)
    return ScanResult(
        passed=len(violations) == 0,
        violations=violations,
        instruction_density=0.0,
    )


def scan_chunks(chunks: List[Document], source_filename: str = "") -> ScanResult:
    """
    Scan all chunks from a document.

    Raises:
        SecurityViolation — if any hard-fail pattern fires, or if the
                            instruction-density threshold is exceeded.

    Returns:
        ScanResult with passed=True if everything is clean.
    """
    all_violations: List[str] = []
    instruction_chunk_count = 0

    for chunk in chunks:
        text = chunk.page_content

        # Hard-fail patterns
        result = scan_text(text)
        if not result.passed:
            all_violations.extend(
                f"[chunk {chunk.metadata.get('chunk_index', '?')}] {v}"
                for v in result.violations
            )

        # Soft instruction density
        if _INSTRUCTION_LIKE.search(text):
            instruction_chunk_count += 1

    density = instruction_chunk_count / max(len(chunks), 1)
    density_violation = density > _MAX_INSTRUCTION_DENSITY

    if all_violations or density_violation:
        details = all_violations.copy()
        if density_violation:
            details.append(
                f"instruction-like density too high "
                f"({density:.0%} of chunks, limit {_MAX_INSTRUCTION_DENSITY:.0%})"
            )
        label = f" in '{source_filename}'" if source_filename else ""
        raise SecurityViolation(
            f"Document{label} failed security scan — possible prompt-injection "
            f"payload detected:\n" + "\n".join(f"  • {d}" for d in details)
        )

    return ScanResult(passed=True, violations=[], instruction_density=density)


def sanitize_query(query: str) -> str:
    """
    Light sanitisation of the user query itself.

    - Strips zero-width characters.
    - Truncates to 2 000 characters to prevent context-flooding.
    - Raises SecurityViolation for hard-fail patterns.
    """
    # Strip zero-width / direction characters
    query = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]", "", query)
    query = query[:2000]

    result = scan_text(query)
    if not result.passed:
        raise SecurityViolation(
            "Query contains potentially unsafe content: "
            + ", ".join(result.violations)
        )
    return query
