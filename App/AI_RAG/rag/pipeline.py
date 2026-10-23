"""NourFi RAG pipeline v2

Changes vs v1
- 1 LLM call per question (no LLM planner, no per-request summary, no citation-repair call)
- Rate limits: Retry-After header, fallback model, concurrency semaphore, no global sleep
- Citations mapped in code: the model cites [1], [2]; we convert to [Source : AAOIFI Standard X]
- Prompt allows partial answers instead of defaulting to "insufficient"
- Query cleaning (greetings, typos), smarter follow-up handling, broad-question retrieval
- Optional similarity gate (MIN_SIM) after retrieval, off until you calibrate it
- DB connection pool, answer cache, per-request logging
"""

from __future__ import annotations

import argparse
import difflib
import os
import random
import re
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import numpy as np
import psycopg
import torch
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging
from huggingface_hub import logging as huggingface_logging

try:
    from psycopg_pool import ConnectionPool
except ImportError:  # pip install "psycopg[pool]" to enable pooling
    ConnectionPool = None

# Same project-root trick as your original file (pipeline.py sits 3 levels below the root),
# so that "from App.prompts.personas import ..." resolves however the app is started.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from App.prompts.personas import PERSONA_PROMPTS
except ImportError:
    try:
        from App.App.prompts.personas import PERSONA_PROMPTS
    except ImportError:
        PERSONA_PROMPTS = {}

# Settings live in config.py, wording/patterns in prompts.py. They are imported as a package when
# the backend loads this file, or as plain modules when it is run from its own folder (calibrate.py).
if __package__:
    from .config import (
        resolve_database_url,
        DB_POOL_MAX,
        DB_POOL_MAX_IDLE,
        DB_POOL_MAX_LIFETIME,
        DB_CONNECT_TIMEOUT,
        EMBEDDING_MODEL,
        MAX_QUERY_LENGTH,
        TOP_K,
        CANDIDATE_MULTIPLIER,
        CANDIDATE_MIN,
        MAX_CHUNK_CHARS,
        MIN_SIM,
        CORE_STANDARDS,
        BROAD_MIN_STANDARDS,
        TRANSLATE_ARABIC,
        TYPO_CUTOFF,
        TYPO_MIN_LENGTH,
        GROQ_API_KEY,
        GROQ_BASE_URL,
        GROQ_MODEL,
        FALLBACK_MODEL,
        REASONING_EFFORT,
        SUMMARY_MODEL,
        SECOND_API_KEY,
        SECOND_BASE_URL,
        SECOND_MODEL,
        LLM_REQUEST_TIMEOUT,
        GENERATION_TEMPERATURE,
        MAX_GENERATION_TOKENS,
        RETRY_MAX_GENERATION_TOKENS,
        MAX_RETRIES,
        LLM_MAX_CONCURRENCY,
        MAX_WAIT_SECONDS,
        SHORT_WAIT_SECONDS,
        LLM_TPM_LIMIT,
        TPM_MAX_QUEUE_SECONDS,
        TPM_WINDOW_SECONDS,
        SECOND_COOLDOWN_SECONDS,
        TRANSLATE_MAX_TOKENS,
        MEMORY_ANSWER_MAX_TOKENS,
        SUMMARY_CALL_MAX_TOKENS,
        SUMMARY_TEMPERATURE,
        MODEL_CONTEXT_TOKENS,
        PROMPT_TOKEN_MARGIN,
        PROMPT_TOKEN_BUDGET,
        CHARS_PER_TOKEN,
        KEEP_RECENT_TURNS,
        HISTORY_ASSISTANT_MAX_CHARS,
        SUMMARY_MAX_TOKENS,
        CROSS_CONTEXT_MAX_TOKENS,
        CACHE_TTL_SECONDS,
        CACHE_MAX_ENTRIES,
    )
    from .prompts import (
        CITATION_RE,
        SENSITIVE_NUMBER_RE,
        DISCLAIMER,
        INSUFFICIENT,
        FATWA_REPLY,
        GREETING_REPLY,
        EMPTY_REPLY,
        FOLLOWUP_HELP,
        CANNED_PREFIXES,
        SYSTEM_PROMPT,
        TRANSLATE_SYSTEM_PROMPT,
        MEMORY_SYSTEM_PROMPT,
        SUMMARY_SYSTEM_PROMPT,
        PERSONA_TEMPLATE,
        CROSS_CONTEXT_TEMPLATE,
        SUMMARY_BLOCK_TEMPLATE,
        EXCERPTS_HEADER,
        ANSWER_INSTRUCTION,
        NUMBER_WARNING_TEMPLATE,
        BLOCKED_REASONING_MARKERS,
        FATWA_DIRECT_PATTERNS,
        FRAMEWORK_RE,
        GREETING_WORDS,
        GREETING_PREFIX_RE,
        TYPO_MAP,
        DOMAIN_VOCAB,
        BROAD_RE,
        BROAD_DOMAIN_RE,
        PRONOUN_RE,
        FOLLOWUP_START_RE,
        MEMORY_RE,
        NO_MEMORY_REPLY,
        NO_MEMORY_REPLY_AR,
        STANDARD_RES,
    )
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        resolve_database_url,
        DB_POOL_MAX,
        DB_POOL_MAX_IDLE,
        DB_POOL_MAX_LIFETIME,
        DB_CONNECT_TIMEOUT,
        EMBEDDING_MODEL,
        MAX_QUERY_LENGTH,
        TOP_K,
        CANDIDATE_MULTIPLIER,
        CANDIDATE_MIN,
        MAX_CHUNK_CHARS,
        MIN_SIM,
        CORE_STANDARDS,
        BROAD_MIN_STANDARDS,
        TRANSLATE_ARABIC,
        TYPO_CUTOFF,
        TYPO_MIN_LENGTH,
        GROQ_API_KEY,
        GROQ_BASE_URL,
        GROQ_MODEL,
        FALLBACK_MODEL,
        REASONING_EFFORT,
        SUMMARY_MODEL,
        SECOND_API_KEY,
        SECOND_BASE_URL,
        SECOND_MODEL,
        LLM_REQUEST_TIMEOUT,
        GENERATION_TEMPERATURE,
        MAX_GENERATION_TOKENS,
        RETRY_MAX_GENERATION_TOKENS,
        MAX_RETRIES,
        LLM_MAX_CONCURRENCY,
        MAX_WAIT_SECONDS,
        SHORT_WAIT_SECONDS,
        LLM_TPM_LIMIT,
        TPM_MAX_QUEUE_SECONDS,
        TPM_WINDOW_SECONDS,
        SECOND_COOLDOWN_SECONDS,
        TRANSLATE_MAX_TOKENS,
        MEMORY_ANSWER_MAX_TOKENS,
        SUMMARY_CALL_MAX_TOKENS,
        SUMMARY_TEMPERATURE,
        MODEL_CONTEXT_TOKENS,
        PROMPT_TOKEN_MARGIN,
        PROMPT_TOKEN_BUDGET,
        CHARS_PER_TOKEN,
        KEEP_RECENT_TURNS,
        HISTORY_ASSISTANT_MAX_CHARS,
        SUMMARY_MAX_TOKENS,
        CROSS_CONTEXT_MAX_TOKENS,
        CACHE_TTL_SECONDS,
        CACHE_MAX_ENTRIES,
    )
    from prompts import (
        CITATION_RE,
        SENSITIVE_NUMBER_RE,
        DISCLAIMER,
        INSUFFICIENT,
        FATWA_REPLY,
        GREETING_REPLY,
        EMPTY_REPLY,
        FOLLOWUP_HELP,
        CANNED_PREFIXES,
        SYSTEM_PROMPT,
        TRANSLATE_SYSTEM_PROMPT,
        MEMORY_SYSTEM_PROMPT,
        SUMMARY_SYSTEM_PROMPT,
        PERSONA_TEMPLATE,
        CROSS_CONTEXT_TEMPLATE,
        SUMMARY_BLOCK_TEMPLATE,
        EXCERPTS_HEADER,
        ANSWER_INSTRUCTION,
        NUMBER_WARNING_TEMPLATE,
        BLOCKED_REASONING_MARKERS,
        FATWA_DIRECT_PATTERNS,
        FRAMEWORK_RE,
        GREETING_WORDS,
        GREETING_PREFIX_RE,
        TYPO_MAP,
        DOMAIN_VOCAB,
        BROAD_RE,
        BROAD_DOMAIN_RE,
        PRONOUN_RE,
        FOLLOWUP_START_RE,
        MEMORY_RE,
        NO_MEMORY_REPLY,
        NO_MEMORY_REPLY_AR,
        STANDARD_RES,
    )

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()
huggingface_logging.set_verbosity_error()



def _log(message: str) -> None:
    print(f"[nourfi] {message}", file=sys.stderr)


# -----------------------------------------
# SAFETY / FATWA
# -----------------------------------------


def _fold(text: str) -> str:
    q = " ".join(text.lower().split())
    for src, dst in (
        ("é", "e"), ("è", "e"), ("ê", "e"), ("à", "a"), ("â", "a"), ("ç", "c"),
        ("û", "u"), ("î", "i"), ("ô", "o"), ("ï", "i"), ("ü", "u"),
        ("؟", "?"), ("،", ","), ("’", "'"), ("`", "'"),
    ):
        q = q.replace(src, dst)
    return q


def is_fatwa_request(question: str) -> bool:
    if not question:
        return False
    q = _fold(question)
    if not any(p in q for p in FATWA_DIRECT_PATTERNS):
        return False
    # "how/conditions/criteria" questions ask about a framework, not a ruling.
    # Word boundaries so that "show" no longer matches "how".
    return not FRAMEWORK_RE.search(q)


def redact_sensitive_data(text: str) -> str:
    if not text:
        return text
    return SENSITIVE_NUMBER_RE.sub("[REDACTED_NUMBER]", text)


def clean_answer(answer: str) -> str:
    """Cut the answer if the model leaks a reasoning section (marker at the START of a line)."""
    kept = []
    for line in answer.splitlines():
        normalized = line.strip().lower().lstrip("#*- ")
        if normalized.startswith(BLOCKED_REASONING_MARKERS):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def trim_to_last_sentence(text: str) -> str:
    """For answers cut by max_tokens: drop the unfinished trailing sentence."""
    text = text.rstrip()
    if not text or text[-1] in ".!?:)]»\"'":
        return text
    cut = max(text.rfind(". "), text.rfind(".\n"), text.rfind("! "), text.rfind("? "))
    return text[: cut + 1].rstrip() if cut > len(text) * 0.4 else text


# -----------------------------------------
# CITATIONS (mapped in code)
# -----------------------------------------
_NUM_CITE_RE = re.compile(  # [1]  [1, 2]  [SOURCE 1]  and the 【1】 / ［1］ forms some models emit
    r"[\[【［](?:SOURCE\s*)?(\d{1,2}(?:\s*[,;，]\s*(?:SOURCE\s*)?\d{1,2})*)[\]】］]", re.IGNORECASE
)
_DUP_CITE_RE = re.compile(
    r"(\[Source : AAOIFI Standard (\d+)\])(?:\s*\[Source : AAOIFI Standard \2\])+"
)


_NUM_RE = re.compile(r"\d[\d\u00a0\u202f ,.]*\d|\d")
_SEP_RE = re.compile(r"[\s,.\u00a0\u202f]")


def find_ungrounded_numbers(answer: str, question: str, results: list[dict[str, Any]]) -> list[str]:
    """Numbers in the answer that appear neither in the excerpts the model saw nor in the question."""
    body = CITATION_RE.sub(" ", answer)
    body = re.sub(r"(?m)^\s*\d+[.)]\s", " ", body)  # "1. " list markers
    seen = " ".join(
        f"{(r.get('content') or '')[:MAX_CHUNK_CHARS]} {r.get('standard_number', '')} "
        f"{r.get('page_start', '')} {r.get('page_end', '')}"
        for r in results
    )
    known = _SEP_RE.sub("", seen + " " + question)
    bad: list[str] = []
    for m in _NUM_RE.finditer(body):
        raw = m.group(0).strip(" ,.")
        digits = _SEP_RE.sub("", raw)
        if not digits.isdigit() or (len(raw) == len(digits) and int(digits) <= 10):
            continue
        if digits not in known:
            bad.append(raw)
    return list(dict.fromkeys(bad))


def add_number_warning(answer: str, question: str, results: list[dict[str, Any]]) -> str:
    bad = find_ungrounded_numbers(answer, question, results)
    if not bad:
        return answer
    _log(f"UNGROUNDED NUMBERS in answer: {bad}")
    shown = ", ".join(bad[:4])
    return f"{answer}\n\n{NUMBER_WARNING_TEMPLATE.format(shown=shown)}"


def _drop_redundant_trailing_citation(text: str) -> str:
    """Remove a last paragraph that contains ONLY citations already given earlier in the answer."""
    paras = re.split(r"\n\s*\n", text.strip())
    if len(paras) < 2:
        return text
    last = paras[-1].strip()
    stds = set(CITATION_RE.findall(last))
    if stds and not CITATION_RE.sub("", last).strip(" *_.:-\n"):
        if stds <= set(CITATION_RE.findall("\n\n".join(paras[:-1]))):
            return "\n\n".join(paras[:-1]).strip()
    return text


def map_citations(answer: str, results: list[dict[str, Any]]) -> str:
    """[1] / [2][3] / [1, 2] -> [Source : AAOIFI Standard X]. Out-of-range numbers are dropped."""

    def repl(match: re.Match) -> str:
        standards: list[str] = []
        for num in re.findall(r"\d+", match.group(1)):
            idx = int(num) - 1
            if 0 <= idx < len(results):
                std = str(results[idx].get("standard_number", "")).strip()
                if std and std not in standards:
                    standards.append(std)
        return " ".join(f"[Source : AAOIFI Standard {s}]" for s in standards)

    mapped = _NUM_CITE_RE.sub(repl, answer)
    # Remove (instead of rejecting the whole answer) any citation to a standard we did not retrieve.
    allowed = {str(r.get("standard_number", "")).strip() for r in results}
    mapped = CITATION_RE.sub(lambda m: m.group(0) if m.group(1) in allowed else "", mapped)
    mapped = _DUP_CITE_RE.sub(r"\1", mapped)  # [Std 13][Std 13][Std 13] -> [Std 13]
    mapped = re.sub(r"(?<=[^\s\]])(\[Source : AAOIFI Standard)", r" \1", mapped)  # "value[Source" -> "value [Source"
    mapped = _drop_redundant_trailing_citation(mapped)
    mapped = re.sub(r"[ \t]+\n", "\n", mapped)
    return re.sub(r"(?<=\S)[ ]{2,}", " ", mapped).strip()


# -----------------------------------------
# QUERY CLEANING / ROUTING (rule-based, no LLM)
# -----------------------------------------


def smalltalk_reply(question: str) -> str | None:
    tokens = re.findall(r"[\w']+", question.lower())
    if tokens and len(tokens) <= 5 and all(t in GREETING_WORDS for t in tokens):
        return GREETING_REPLY
    return None




def is_memory_question(question: str) -> bool:
    """Questions about the conversation itself ('summarize our previous conversations'). They must
    NOT go through AAOIFI retrieval: unrelated excerpts would be presented as 'what we discussed'."""
    return bool(MEMORY_RE.search(question))


def answer_from_memory(
    question: str,
    conversation_summary: str | None,
    recent_messages: list[dict[str, str]] | None,
    cross_conversation_context: str | None,
    model_name: str,
) -> str:
    cross = _cap_tokens(redact_sensitive_data(cross_conversation_context or ""), CROSS_CONTEXT_MAX_TOKENS)
    summary = _cap_tokens(conversation_summary or "", SUMMARY_MAX_TOKENS)
    turns = [
        f"{'Assistant' if m.get('role') == 'assistant' else 'User'}: "
        f"{_compact_history_turn(m.get('role', 'user'), m.get('content', ''))}"
        for m in (recent_messages or [])[-KEEP_RECENT_TURNS:]
    ]
    if not (cross or summary or turns):
        return NO_MEMORY_REPLY_AR if has_arabic(question) else NO_MEMORY_REPLY
    notes = (
        (f"Earlier conversations:\n{cross}\n\n" if cross else "")
        + (f"This conversation, summary:\n{summary}\n\n" if summary else "")
        + (("This conversation, latest messages:\n" + "\n".join(turns)) if turns else "")
    )
    response = call_llm_with_retries(
        create_client(),
        model=model_name,
        messages=[
            {"role": "system", "content": MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Notes:\n{notes}\n\nQuestion: {redact_sensitive_data(question)}"},
        ],
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MEMORY_ANSWER_MAX_TOKENS,
    )
    text = (response.choices[0].message.content or "").strip()
    return clean_answer(text) if text else (NO_MEMORY_REPLY_AR if has_arabic(question) else NO_MEMORY_REPLY)


def clean_query(question: str) -> str:
    """Strip leading greetings and fix common typos before embedding."""
    q = GREETING_PREFIX_RE.sub("", question).strip() or question.strip()
    q = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", q)  # "alll" -> "all", "pleeease" -> "pleease"

    def fix(match: re.Match) -> str:
        token = match.group(0)
        low = token.lower()
        if low in TYPO_MAP:
            return TYPO_MAP[low]
        if len(low) >= TYPO_MIN_LENGTH and low not in DOMAIN_VOCAB:
            close = [
                w for w in difflib.get_close_matches(low, DOMAIN_VOCAB, n=1, cutoff=TYPO_CUTOFF)
                if abs(len(w) - len(low)) <= 2  # "partners" must NOT become "partnership"
            ]
            if close:
                return close[0]
        return token

    return re.sub(r"[A-Za-z]+", fix, q)


_AR_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


def has_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if _AR_CHAR_RE.match(c)) / len(letters) > 0.3


def translate_to_english(text: str, model_name: str | None = None) -> str:
    """The corpus is English and e5-small is weak Arabic->English, so Arabic questions are
    translated before embedding (1 short extra LLM call, Arabic questions only)."""
    try:
        response = call_llm_with_retries(
            create_client(),
            model=model_name or GROQ_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=TRANSLATE_MAX_TOKENS,
        )
        out = (response.choices[0].message.content or "").strip()
        return out or text
    except Exception as e:  # noqa: BLE001 - never fail the whole request for this
        _log(f"translation failed, using the original query: {type(e).__name__}: {str(e)[:200]}")
        return text


def plan_retrieval(
    question: str, history: list[dict[str, str]] | None = None, model_name: str | None = None
) -> dict[str, str]:
    """Everything that decides WHAT is searched: cleaning, follow-up merge, Arabic translation,
    broad/normal mode. Shared by answer_question and calibrate.py so they behave the same."""
    cleaned = clean_query(question)
    query = build_retrieval_query(cleaned, history or [])
    embed_text, match_text = query, query
    if TRANSLATE_ARABIC and has_arabic(query):
        translated = translate_to_english(query, model_name)
        _log(f"translated query: {translated!r}")
        embed_text, match_text = translated, f"{translated} {query}"
    mode = "broad" if is_broad(match_text) else "normal"
    return {"embed_text": embed_text, "match_text": match_text, "mode": mode}


def is_broad(question: str) -> bool:
    return (
        bool(BROAD_RE.search(question))
        and bool(BROAD_DOMAIN_RE.search(question))
        and not preferred_standards(question)
    )


def is_followup(question: str) -> bool:
    if preferred_standards(question):
        return False
    words = question.split()
    if len(words) <= 4:
        return True
    if len(words) <= 12 and PRONOUN_RE.search(question):
        return True
    return len(words) <= 6 and bool(FOLLOWUP_START_RE.search(question))


def is_canned(text: str) -> bool:
    return (text or "").strip().startswith(CANNED_PREFIXES)


def clean_history(recent_messages: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Drop empty messages and our own canned/refusal replies so they are not fed back to the model."""
    out = []
    for m in recent_messages or []:
        content = (m.get("content") or "").strip()
        if content and not is_canned(content):
            out.append({"role": m.get("role", "user"), "content": content})
    return out


def build_retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """Prepend the previous user question ONLY for real follow-ups."""
    if not history or not is_followup(question):
        return question
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    return f"{user_turns[-1]} {question}".strip() if user_turns else question


# -----------------------------------------
# LLM CALLS: concurrency limit, retry, fallback model
# -----------------------------------------
_llm_slots = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)


def suggested_wait_seconds(error_text: str, fallback: float | None = None) -> float | None:
    """Parse 'Please try again in 1.5s' / '250ms' / '12m5.2s' from a Groq error message."""
    m = re.search(r"try again in\s+([0-9hms.]+)", error_text, re.IGNORECASE)
    if not m:
        return fallback
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    found = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1))
    if not found:
        return fallback
    return sum(float(v) * units[u] for v, u in found) + 0.5


def _retry_wait(error: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if waiting is pointless (e.g. daily quota)."""
    wait = None
    response = getattr(error, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                wait = float(header) + 0.5
            except ValueError:
                pass
    if wait is None:
        wait = suggested_wait_seconds(str(error))
    if wait is None:
        wait = (2 ** attempt) + random.uniform(0.2, 1.0)
    return None if wait > MAX_WAIT_SECONDS else wait


def _apply_model_options(kwargs: dict[str, Any], primary: bool = True) -> None:
    """reasoning_effort is only sent to gpt-oss models on the primary (Groq) provider."""
    extra = dict(kwargs.get("extra_body") or {})
    if primary and "gpt-oss" in str(kwargs.get("model", "")):
        extra["reasoning_effort"] = REASONING_EFFORT
    else:
        extra.pop("reasoning_effort", None)
    if extra:
        kwargs["extra_body"] = extra
    else:
        kwargs.pop("extra_body", None)


_tpm_log: list[list[float]] = []  # [timestamp, tokens] entries of the last 60 s
_tpm_lock = threading.Lock()
_second_disabled_until = 0.0
_fallback_broken = False  # set when the fallback model does not exist for this account


def _estimate_request_tokens(kwargs: dict[str, Any]) -> int:
    text = "".join(str(m.get("content", "")) for m in kwargs.get("messages", []))
    return int(len(text) / CHARS_PER_TOKEN) + int(kwargs.get("max_tokens") or 0)


def _tpm_reserve(estimate: int) -> list[float] | None:
    """Wait (at most TPM_MAX_QUEUE_SECONDS) until the sliding 60 s window has room for this request,
    so users are queued instead of getting a 429 'busy' error."""
    if not LLM_TPM_LIMIT or estimate >= LLM_TPM_LIMIT:
        return None
    deadline = time.monotonic() + TPM_MAX_QUEUE_SECONDS
    while True:
        now = time.monotonic()
        with _tpm_lock:
            while _tpm_log and now - _tpm_log[0][0] > TPM_WINDOW_SECONDS:
                _tpm_log.pop(0)
            used = sum(t for _, t in _tpm_log)
            if used + estimate <= LLM_TPM_LIMIT or not _tpm_log or now >= deadline:
                entry = [now, float(estimate)]
                _tpm_log.append(entry)
                return entry
            wait = _tpm_log[0][0] + TPM_WINDOW_SECONDS - now + 0.2
        _log(f"TPM queue: {used:.0f}/{LLM_TPM_LIMIT} tokens used in the last minute, waiting {wait:.1f}s")
        time.sleep(min(max(wait, 0.2), max(deadline - now, 0.2)))


def _disable_second() -> None:
    global _second_disabled_until
    _second_disabled_until = time.monotonic() + SECOND_COOLDOWN_SECONDS
    _log(f"second provider disabled for {SECOND_COOLDOWN_SECONDS:.0f}s (it failed) - using the primary only")


def call_llm_with_retries(client: Any, *, max_retries: int = MAX_RETRIES, **kwargs):
    """Order of defence when the main call fails:
    0) client-side TPM queue (avoids most 429 in the first place)
    1) short 429 (< SHORT_WAIT_SECONDS): wait and retry the same model
    2) second provider (independent quota), if SECOND_* is set and healthy
    3) fallback model on the same provider, on 429
    4) wait (Retry-After) and retry
    A failing second provider is parked for a few minutes and the primary is used again."""
    kwargs = dict(kwargs)
    primary_model = kwargs.get("model")
    active, on_primary = client, True
    second = create_second_client() if time.monotonic() >= _second_disabled_until else None
    last_error: Exception | None = None
    for attempt in range(max_retries):
        _apply_model_options(kwargs, primary=on_primary)
        entry = None
        if on_primary and kwargs.get("model") == primary_model:
            entry = _tpm_reserve(_estimate_request_tokens(kwargs))
        try:
            with _llm_slots:
                response = active.chat.completions.create(**kwargs)
            if entry is not None:
                used = getattr(getattr(response, "usage", None), "total_tokens", None)
                if used:
                    entry[1] = float(used)  # replace the estimate with the real usage
            return response
        except Exception as e:  # noqa: BLE001
            if entry is not None:
                entry[1] = 0.0  # the provider rejected it: nothing was consumed
            last_error = e
            retryable = isinstance(
                e, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
            )
            _log(f"LLM error ({kwargs.get('model')}): {type(e).__name__}: {str(e)[:300]}")

            if kwargs.get("model") == FALLBACK_MODEL and primary_model != FALLBACK_MODEL and on_primary:
                text = str(e).lower()
                if "model_not_found" in text or "does not exist" in text or "decommission" in text:
                    global _fallback_broken
                    _fallback_broken = True
                    _log(f"fallback model {FALLBACK_MODEL} is unavailable for this account - disabled; set GROQ_FALLBACK_MODEL")
                kwargs["model"] = primary_model
                if attempt == max_retries - 1:
                    break
                if _fallback_broken or not retryable:
                    break  # nothing else left to try on this provider
                continue

            if not on_primary:  # the second provider failed: never let it make things worse
                _disable_second()
                second, active, on_primary = None, client, True
                kwargs["model"] = primary_model
                if attempt == max_retries - 1:
                    break
                continue

            if attempt == max_retries - 1:
                break
            if not retryable:  # 400/401/413...: waiting will not help, only another provider can
                if second is not None:
                    _log(f"switching to second provider ({SECOND_MODEL})")
                    active, on_primary, kwargs["model"] = second, False, SECOND_MODEL
                    continue
                raise

            too_large = "request too large" in str(e).lower()
            wait = _retry_wait(e, attempt)
            if isinstance(e, RateLimitError) and wait is not None and wait <= SHORT_WAIT_SECONDS and not too_large:
                _log(f"short rate limit, waiting {wait:.1f}s on the same model")
                time.sleep(wait)
                continue
            if second is not None:
                _log(f"switching to second provider ({SECOND_MODEL})")
                active, on_primary, kwargs["model"] = second, False, SECOND_MODEL
                continue
            if too_large:
                break  # prompt + max_tokens exceeds the limit itself
            if (
                isinstance(e, RateLimitError)
                and FALLBACK_MODEL
                and not _fallback_broken
                and kwargs.get("model") != FALLBACK_MODEL
            ):
                _log(f"429 on {kwargs.get('model')}, switching to {FALLBACK_MODEL}")
                kwargs["model"] = FALLBACK_MODEL
                time.sleep(0.5)
                continue
            if wait is None:
                break
            _log(f"retry {attempt + 1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("Unknown LLM error")


@lru_cache(maxsize=1)
def create_second_client() -> OpenAI | None:
    """Optional second provider (any OpenAI-compatible API). None if not configured."""
    if not (SECOND_API_KEY and SECOND_BASE_URL and SECOND_MODEL):
        return None
    return OpenAI(api_key=SECOND_API_KEY, base_url=SECOND_BASE_URL, timeout=LLM_REQUEST_TIMEOUT, max_retries=0)


@lru_cache(maxsize=1)
def create_client() -> OpenAI:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in .env")
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL, timeout=LLM_REQUEST_TIMEOUT, max_retries=0)


create_groq_client = create_client  # backwards compatibility


# -----------------------------------------
# DATABASE
# -----------------------------------------
_pool = None
_pool_lock = threading.Lock()


@contextmanager
def db_connection(database_url: str):
    """Pooled connection if psycopg_pool is installed, otherwise a plain connection.
    prepare_threshold=None keeps it safe behind Supabase's pgbouncer pooler."""
    global _pool
    if ConnectionPool is not None:
        with _pool_lock:
            if _pool is None:
                extra = {}
                if hasattr(ConnectionPool, "check_connection"):  # psycopg_pool >= 3.2
                    extra["check"] = ConnectionPool.check_connection  # drop dead connections
                _pool = ConnectionPool(
                    database_url,
                    min_size=1,
                    max_size=DB_POOL_MAX,
                    max_idle=DB_POOL_MAX_IDLE,
                    max_lifetime=DB_POOL_MAX_LIFETIME,
                    kwargs={"connect_timeout": DB_CONNECT_TIMEOUT, "prepare_threshold": None},
                    open=True,
                    **extra,
                )
        with _pool.connection() as conn:
            yield conn
    else:
        with psycopg.connect(database_url, connect_timeout=DB_CONNECT_TIMEOUT, prepare_threshold=None) as conn:
            yield conn


# -----------------------------------------
# EMBEDDING + RETRIEVAL
# -----------------------------------------
def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


@lru_cache(maxsize=1)
def load_encoder() -> tuple[Any, Any, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    encoder = AutoModel.from_pretrained(EMBEDDING_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()
    return tokenizer, encoder, device


def embed_query(question: str, tokenizer: Any, encoder: Any, device: torch.device) -> np.ndarray:
    encoded = tokenizer(
        ["query: " + question.strip()],
        padding=True,
        truncation=True,
        max_length=MAX_QUERY_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        output = encoder(**encoded)
    hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0].astype(np.float32)




def preferred_standards(question: str) -> set[str]:
    return {std for rx, std in STANDARD_RES if rx.search(question)}


_COLS = (
    "chunk_id, standard_number, standard_title, section, page_start, page_end, content, "
    "1 - (embedding <=> %(v)s::vector) as similarity"
)


def _rows(cursor) -> list[dict[str, Any]]:
    cols = [c.name for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def retrieve(
    database_url: str,
    query_vector: np.ndarray,
    question: str,
    mode: str = "normal",
) -> list[dict[str, Any]]:
    vector = vector_literal(query_vector)

    with db_connection(database_url) as conn, conn.cursor() as cur:
        # Broad question ("all the methods..."): best chunk from each core standard.
        if mode == "broad":
            cur.execute(
                f"""
                select * from (
                    select {_COLS},
                           row_number() over (
                               partition by standard_number order by embedding <=> %(v)s::vector
                           ) as rn
                    from public.rag_chunks
                    where standard_number::text = any(%(stds)s::text[])
                ) t
                where rn <= 1
                order by similarity desc
                """,
                {"v": vector, "stds": list(CORE_STANDARDS)},
            )
            rows = _rows(cur)
            for r in rows:
                r.pop("rn", None)
            if len(rows) >= BROAD_MIN_STANDARDS:  # DB really has the core standards
                return rows
            # otherwise fall through to normal retrieval

        candidate_limit = max(TOP_K * CANDIDATE_MULTIPLIER, CANDIDATE_MIN)
        cur.execute(
            f"select {_COLS} from public.rag_chunks "
            "order by embedding <=> %(v)s::vector limit %(n)s",
            {"v": vector, "n": candidate_limit},
        )
        candidates = _rows(cur)

        targets = preferred_standards(question)
        if not targets:
            return candidates[:TOP_K]

        existing = {str(c.get("standard_number", "")).strip() for c in candidates}
        for target in sorted(targets - existing):
            cur.execute(
                f"select {_COLS} from public.rag_chunks "
                "where standard_number::text = %(t)s "
                "order by embedding <=> %(v)s::vector limit %(n)s",
                {"v": vector, "t": target, "n": TOP_K},
            )
            candidates.extend(_rows(cur))

    selected: list[dict[str, Any]] = []
    selected_ids: set[Any] = set()
    for target in sorted(targets):
        for item in candidates:
            if (
                str(item.get("standard_number", "")).strip() == target
                and item.get("chunk_id") not in selected_ids
            ):
                selected.append(item)
                selected_ids.add(item.get("chunk_id"))
                break
    for item in candidates:
        if len(selected) >= TOP_K:
            break
        if item.get("chunk_id") not in selected_ids:
            selected.append(item)
            selected_ids.add(item.get("chunk_id"))
    return selected


def retrieve_with_retry(*args, **kwargs) -> list[dict[str, Any]]:
    """A pooled DB connection can be dead (idle timeout on the server side): retry once."""
    try:
        return retrieve(*args, **kwargs)
    except psycopg.OperationalError as e:
        _log(f"DB connection error, retrying once: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(0.5)
        return retrieve(*args, **kwargs)


def format_context(results: list[dict[str, Any]], max_chars: int | None = None) -> str:
    blocks = []
    for index, item in enumerate(results, 1):
        content = item.get("content", "") or ""
        if max_chars:
            content = content[:max_chars]
        blocks.append(
            f"[SOURCE {index}]\n"
            f"AAOIFI Standard: {item.get('standard_number', '?')}\n"
            f"Title: {item.get('standard_title', '?')}\n"
            + (f"Section: {item['section']}\n" if item.get("section") else "")
            + f"Pages: {item.get('page_start', '?')}-{item.get('page_end', '?')}\n"
            f"Content: {content}"
        )
    return "\n\n".join(blocks)


# -----------------------------------------
# PROMPT BUDGET
# -----------------------------------------
def approx_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _cap_tokens(text: str, max_tokens: int) -> str:
    text = (text or "").strip()
    limit = max(0, max_tokens) * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,;:") + " ..."


def _compact_history_turn(role: str, content: str) -> str:
    """Past assistant answers are replayed as a short gist: no citation tags, no formatting, first
    ~350 chars. The topic is what a follow-up needs, and it saves ~400 tokens per old answer."""
    if role != "assistant":
        return content
    text = CITATION_RE.sub("", content)
    text = re.sub(r"[*_#>|`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > HISTORY_ASSISTANT_MAX_CHARS:
        cut = text[:HISTORY_ASSISTANT_MAX_CHARS]
        text = (cut.rsplit(" ", 1)[0] if " " in cut else cut) + " ..."
    return text


def build_context_blocks(
    persona_instructions: str,
    safe_question: str,
    results: list[dict[str, Any]],
    conversation_summary: str | None,
    recent_messages: list[dict[str, str]] | None,
) -> tuple[str, list[dict[str, str]]]:
    tokens_used = (
        approx_tokens_from_text(SYSTEM_PROMPT)
        + approx_tokens_from_text(persona_instructions)
        + approx_tokens_from_text("Question:\n" + safe_question)
        + approx_tokens_from_text(format_context(results, max_chars=MAX_CHUNK_CHARS))
    )
    allowed_prompt_tokens = min(
        MODEL_CONTEXT_TOKENS - MAX_GENERATION_TOKENS - PROMPT_TOKEN_MARGIN, PROMPT_TOKEN_BUDGET
    )
    if allowed_prompt_tokens <= 0:
        allowed_prompt_tokens = max(256, MODEL_CONTEXT_TOKENS // 8)

    summary_block = ""
    if conversation_summary:
        remaining = allowed_prompt_tokens - tokens_used
        text = _cap_tokens(conversation_summary, SUMMARY_MAX_TOKENS)
        if approx_tokens_from_text(text) > remaining:
            text = text[: max(0, remaining) * CHARS_PER_TOKEN]
            text = text.rsplit(" ", 1)[0] if " " in text else text
        if text:
            summary_block = SUMMARY_BLOCK_TEMPLATE.format(text=text)
            tokens_used += approx_tokens_from_text(text)

    turns: list[dict[str, str]] = []
    for message in reversed((recent_messages or [])[-KEEP_RECENT_TURNS:]):
        role = message.get("role", "user")
        content = _compact_history_turn(role, message.get("content", ""))
        need = approx_tokens_from_text(content)
        if tokens_used + need <= allowed_prompt_tokens:
            turns.insert(0, {"role": role, "content": content})
            tokens_used += need
            continue
        remaining_chars = (allowed_prompt_tokens - tokens_used) * CHARS_PER_TOKEN
        if remaining_chars >= 50:
            trimmed = content[:remaining_chars]
            trimmed = trimmed.rsplit(" ", 1)[0] if " " in trimmed else trimmed
            if trimmed:
                turns.insert(0, {"role": role, "content": trimmed + " ..."})
        break
    return summary_block, turns


def persona_block(persona_name: str) -> str:
    persona_prompt = PERSONA_PROMPTS.get(persona_name, "")
    if not persona_prompt:
        # Visible instead of silent: the persona would otherwise change nothing.
        _log(f"WARNING: no prompt found for persona {persona_name!r} "
             f"(known: {sorted(PERSONA_PROMPTS) or 'none - personas module not imported'})")
    return PERSONA_TEMPLATE.format(name=persona_name, prompt=persona_prompt)


def summarize_conversation(
    conversation_summary: str | None,
    recent_messages: list[dict[str, str]] | None = None,
    model_name: str | None = None,
) -> str:
    """Call this from the backend every ~4 user turns (not on every request) and store the result
    with the conversation; pass it back as conversation_summary. Uses the cheap fallback model
    by default so it does not eat the main model's quota. ~300 tokens per call."""
    if not recent_messages:
        return conversation_summary or ""
    model = model_name or SUMMARY_MODEL or (GROQ_MODEL if _fallback_broken else FALLBACK_MODEL) or GROQ_MODEL
    history = "\n".join(
        f"{m.get('role', 'user').capitalize()}: {_compact_history_turn(m.get('role', 'user'), m.get('content', ''))}"
        for m in recent_messages
    )
    response = call_llm_with_retries(
        create_client(),
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Existing summary:\n{conversation_summary or '(none)'}\n\n"
                    f"New messages:\n{history}"
                ),
            },
        ],
        temperature=SUMMARY_TEMPERATURE,
        max_tokens=SUMMARY_CALL_MAX_TOKENS,
    )
    return _cap_tokens(redact_sensitive_data((response.choices[0].message.content or "").strip()), SUMMARY_MAX_TOKENS)


# -----------------------------------------
# GENERATION (exactly 1 LLM call)
# -----------------------------------------
def generate_answer(
    question: str,
    results: list[dict[str, Any]],
    model_name: str = GROQ_MODEL,
    persona_name: str = "general_public",
    conversation_summary: str | None = None,
    recent_messages: list[dict[str, str]] | None = None,
    cross_conversation_context: str | None = None,
) -> str:
    persona_instructions = persona_block(persona_name)
    safe_question = redact_sensitive_data(question)
    cross_text = _cap_tokens(redact_sensitive_data(cross_conversation_context or ""), CROSS_CONTEXT_MAX_TOKENS)
    cross_block = CROSS_CONTEXT_TEMPLATE.format(text=cross_text) if cross_text else ""
    # the cross-conversation text is counted in the prompt budget together with the persona text
    summary_block, turns = build_context_blocks(
        persona_instructions + cross_block, safe_question, results, conversation_summary, recent_messages
    )
    prompt = (
        f"{persona_instructions}\n{cross_block}{summary_block}"
        f"Question:\n{safe_question}\n\n"
        f"{EXCERPTS_HEADER}\n"
        f"{format_context(results, max_chars=MAX_CHUNK_CHARS)}\n\n"
        f"{ANSWER_INSTRUCTION}"
    )
    messages = [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in turns
    ]
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages,
                     {"role": "user", "content": prompt}]

    _log(f"prompt ~{sum(approx_tokens_from_text(m['content']) for m in full_messages)} tokens, "
         f"max_tokens={MAX_GENERATION_TOKENS}")
    client = create_client()
    response = call_llm_with_retries(
        client, model=model_name, messages=full_messages,
        temperature=GENERATION_TEMPERATURE, max_tokens=MAX_GENERATION_TOKENS,
    )
    choice = response.choices[0]
    answer = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None)

    # Only case that justifies a 2nd call: reasoning ate every token and nothing was written.
    if not answer.strip() and finish == "length":
        _log("empty answer (finish_reason=length), retrying once with a bigger budget")
        response = call_llm_with_retries(
            client, model=model_name, messages=full_messages,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=min(MAX_GENERATION_TOKENS * 2, RETRY_MAX_GENERATION_TOKENS),
        )
        choice = response.choices[0]
        answer = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)

    answer = clean_answer(answer)
    if finish == "length":
        answer = trim_to_last_sentence(answer)
    _log(f"llm finish_reason={finish} model={getattr(response, 'model', model_name)}")

    if not answer.strip():
        _log(f"EMPTY model output (finish_reason={finish}) - technical problem, not 'insufficient'")
        return EMPTY_REPLY
    if answer.strip().lower().startswith(INSUFFICIENT[:40].lower()):
        _log("model itself answered 'insufficient'")
        return INSUFFICIENT

    answer = map_citations(answer, results)
    if not answer:
        return EMPTY_REPLY
    return add_number_warning(answer, safe_question, results)


# -----------------------------------------
# ANSWER CACHE (standalone questions only)
# -----------------------------------------
_cache: "OrderedDict[tuple, tuple[float, tuple[str, list]]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key: tuple):
    with _cache_lock:
        item = _cache.get(key)
        if not item:
            return None
        if time.monotonic() - item[0] > CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return item[1]


def _cache_put(key: tuple, value: tuple[str, list]) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), value)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


# -----------------------------------------
# MAIN ENTRY
# -----------------------------------------
def answer_question(
    question: str,
    database_url: str,
    model_name: str = GROQ_MODEL,
    persona_name: str = "general_public",
    conversation_summary: str | None = None,
    recent_messages: list[dict[str, str]] | None = None,
    cross_conversation_context: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    started = time.monotonic()
    question = redact_sensitive_data(question.strip())
    if not question:
        raise ValueError("The question cannot be empty.")
    if is_fatwa_request(question):
        return FATWA_REPLY, []

    # 1) Greetings / thanks: no retrieval, no LLM.
    canned = smalltalk_reply(question)
    if canned:
        return canned, []

    # 1b) Questions about the conversation itself: answer from memory notes, no AAOIFI retrieval.
    if is_memory_question(question):
        _log("memory question: answering from conversation notes, no retrieval")
        answer = answer_from_memory(
            question, conversation_summary, recent_messages, cross_conversation_context, model_name
        )
        return answer, []

    # 2) Short follow-up ("why", "what") right after a refusal: don't repeat the refusal.
    history = clean_history(recent_messages)
    last_assistant = next(
        (m for m in reversed(recent_messages or []) if m.get("role") == "assistant"), None
    )
    if (
        last_assistant
        and is_canned(last_assistant.get("content", ""))
        and len(question.split()) <= 3
        and is_followup(question)
    ):
        return FOLLOWUP_HELP, []

    # 3) Cache (only for questions that do not depend on conversation context).
    cacheable = not (history or conversation_summary or cross_conversation_context)
    cache_key = (re.sub(r"\W+", " ", clean_query(question).lower()).strip(), persona_name, model_name)
    if cacheable:
        hit = _cache_get(cache_key)
        if hit:
            _log("cache hit")
            return hit

    # 4) Retrieval
    plan = plan_retrieval(question, history, model_name)
    retrieval_query, mode = plan["embed_text"], plan["mode"]
    tokenizer, encoder, device = load_encoder()
    query_vector = embed_query(retrieval_query, tokenizer, encoder, device)
    results = retrieve_with_retry(database_url, query_vector, plan["match_text"], mode)
    best_sim = max((float(r.get("similarity") or 0) for r in results), default=0.0)
    _log(
        f"mode={mode} best_sim={best_sim:.3f} chunks={len(results)} "
        f"standards={[r.get('standard_number') for r in results]} query={retrieval_query!r}"
    )
    if not results:
        return INSUFFICIENT, []
    # Similarity gate (disabled while MIN_SIM=0). Skipped for broad mode, whose chunks are
    # picked per standard and naturally score lower.
    if MIN_SIM and mode == "normal" and best_sim < MIN_SIM:
        _log(f"gate: best_sim {best_sim:.3f} < MIN_SIM {MIN_SIM}")
        return INSUFFICIENT, []

    # 5) One generation call
    answer = generate_answer(
        question, results, model_name, persona_name,
        conversation_summary, history, cross_conversation_context,
    )
    _log(f"done in {time.monotonic() - started:.1f}s")

    out = (answer, results)
    if cacheable and answer not in (INSUFFICIENT, EMPTY_REPLY):
        _cache_put(cache_key, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NourFi AAOIFI RAG pipeline")
    parser.add_argument("--question", help="Question; prompted interactively if omitted")
    parser.add_argument(
        "--database-url",
        default=resolve_database_url()[0],
    )
    parser.add_argument("--model", default=GROQ_MODEL)
    parser.add_argument("--persona", default="general_public")
    args = parser.parse_args()

    question = (args.question or input("Enter your question: ")).strip()
    if not question:
        raise SystemExit("The question cannot be empty.")
    if not args.database_url:
        raise SystemExit("A database URL is required.")

    answer, results = answer_question(
        question, args.database_url, model_name=args.model, persona_name=args.persona
    )
    print(answer)
    print(f"\n{DISCLAIMER}")
    if results:
        print("\nSources:")
        for item in results:
            print(
                f"- AAOIFI Standard {item.get('standard_number', '?')} "
                f"({item.get('standard_title', '?')}, pp. "
                f"{item.get('page_start', '?')}-{item.get('page_end', '?')})"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
    except Exception as error:
        sys.exit(f"Error: {error}")