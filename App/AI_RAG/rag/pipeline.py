"""NourFi RAG: Top-5 AAOIFI retrieval followed by grounded generation
with mandatory strict citations, intent routing, and persona-based style.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import lru_cache
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import psycopg
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging
from huggingface_hub import logging as huggingface_logging

try:
    from App.prompts.personas import PERSONA_PROMPTS
except ImportError:
    PERSONA_PROMPTS = {}

load_dotenv()

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()
huggingface_logging.set_verbosity_error()

MODEL_NAME = os.getenv("MODEL_NAME_3", "intfloat/multilingual-e5-small")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 3     
MAX_QUERY_LENGTH = 128
MAX_CHUNK_CHARS = 1200    
MAX_GENERATION_TOKENS = min(int(os.getenv("MAX_GENERATION_TOKENS", "600")), 900)
MAX_RETRIES = 3

CITATION_RE = re.compile(r"\[Source\s*:\s*AAOIFI Standard ([0-9]+)\]")

GREETING_TERMS = (
    "hello", "hi", "hey", "bonjour", "salut", "coucou", "salem", "salam",
    "how are you", "how r u", "how are u", "comment vas-tu", "comment allez-vous",
    "ca va", "ça va", "good morning", "good evening", "who are you", "what can you do",
    "thanks", "thank you", "merci", "ok", "got it"
)

PLATFORM_TERMS = (
    "conv", "convs", "conversation", "conversations", "chat", "chats",
    "history", "limit", "open", "message", "messages", "account", "profile"
)

META_FORMATTING_TERMS = (
    "explain in detail", "give examples", "more details", "in details",
    "give exmples", "explain simply", "bullet points", "summarize",
    "donne des exemples", "explique en détail", "simplifie"
)

FATWA_TERMS = (
    "fatwa", "give me a fatwa", "issue a fatwa", "is this halal",
    "is this haram", "halal or haram", "فتوى", "حلال", "حرام",
    "halal ou haram", "haram ou halal", "est-ce halal", "est-ce haram",
    "c'est halal", "c'est haram", "est halal", "est haram",
    "licite ou illicite", "permis ou interdit", "autorisé ou interdit",
    "est-ce permis", "est-ce interdit", "est-ce autorisé"
)

MEMORY_TERMS = (
    "memory", "prev conversations", "previous conversations", "past conversations",
    "past chats", "prev chats", "what did we talk about", "remember", "do you remember",
    "what were my last topics", "recall", "historique", "ce qu'on a dit", "mes anciennes conversations"
)

SENSITIVE_NUMBER_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

DISCLAIMER = os.getenv(
    "DISCLAIMER_3",
    "Disclaimer: This answer is based only on retrieved AAOIFI material. "
    "It is for educational purposes and is not a fatwa or a substitute "
    "for advice from a qualified Shariah scholar or legal professional.",
)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT_3",
    """You are Nourfi, an AI financial assistant and neutral Islamic-finance research assistant.

CASUAL GREETINGS & PLATFORM QUESTIONS RULE:
If the user provides a casual greeting, small talk, or platform usage questions (e.g. conversation limits, profile settings), respond politely in 1–2 sentences without searching document excerpts or outputting Shariah citations.

MEMORY & PAST CONVERSATIONS RULE:
Use provided conversation summaries and history for continuity. However, domain answers must still be derived strictly from retrieved AAOIFI excerpts.

DOMAIN & RETRIEVAL RULE:
For domain-specific questions, answer only from the retrieved AAOIFI context. Never invent facts, rulings, or citations. If context is insufficient, say exactly:
The retrieved AAOIFI excerpts are insufficient to answer this question.

STRICT CITATION RULE:
Use ONLY this exact citation format at the end of supported sections:
[Source : AAOIFI Standard X]
"""
)

def classify_intent(question: str) -> str:
    """Classifies user input to route execution path safely."""
    normalized = " ".join(question.lower().split()).strip("?!.,")
    
    if any(term in normalized for term in MEMORY_TERMS):
        return "MEMORY"
    if any(term in normalized for term in FATWA_TERMS):
        return "FATWA"
    if any(phrase in normalized for phrase in GREETING_TERMS) and len(normalized.split()) <= 6:
        return "CHITCHAT"
    if any(term in normalized for term in PLATFORM_TERMS):
        return "PLATFORM"
    if any(phrase in normalized for phrase in META_FORMATTING_TERMS):
        return "META_FORMATTING"
        
    return "AAOIFI_DOMAIN"

def redact_sensitive_data(text: str) -> str:
    return SENSITIVE_NUMBER_RE.sub("[REDACTED_NUMBER]", text)

def clean_answer(answer: str) -> str:
    blocked_markers = (
        "user safety:", "safety categories:", "thinking process:",
        "chain of thought", "internal reasoning", "analysis:",
        "here's a thinking process:", "thinking:", "analysis:",
        "analyze user input:", "identify constraints:",
        "evaluate context availability:",
    )
    kept: list[str] = []
    for line in answer.splitlines():
        normalized = line.strip().lower()
        if any(marker in normalized for marker in blocked_markers):
            break
        kept.append(line)
    return "\n".join(kept).strip()

def normalize_citations(answer: str) -> str:
    paragraphs = re.split(r"\n\s*\n", answer.strip())
    normalized: list[str] = []
    for paragraph in paragraphs:
        matches = list(CITATION_RE.finditer(paragraph))
        if len(matches) > 1:
            last = matches[-1]
            prefix = paragraph[:matches[0].start()]
            suffix = paragraph[last.start():]
            paragraph = prefix.rstrip() + " " + suffix
        normalized.append(paragraph.strip())
    return "\n\n".join(normalized).strip()

def valid_citations(answer: str, results: list[dict[str, Any]]) -> bool:
    allowed = {str(item.get("standard_number", "")).strip() for item in results}
    found = CITATION_RE.findall(answer)
    return bool(found) and all(value.strip() in allowed for value in found)

def fallback_answer(results: list[dict[str, Any]]) -> str:
    return "The retrieved AAOIFI excerpts are insufficient to answer this question."

def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"

@lru_cache(maxsize=1)
def load_encoder() -> tuple[Any, Any, torch.device]:
    print("Loading the embedding model...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()
    return tokenizer, encoder, device

def embed_query(question: str, tokenizer: Any, encoder: Any, device: torch.device) -> np.ndarray:
    encoded = tokenizer(
        ["query: " + question.strip()],
        padding=True, truncation=True, max_length=MAX_QUERY_LENGTH, return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = encoder(**encoded)
    hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0].astype(np.float32)

def preferred_standards(question: str) -> set[str]:
    normalized = question.lower()
    mappings = (
        (("murabaha", "murabahah", "mourabaha", "mourabahah"), "8"),
        (("mudaraba", "mudarabah", "moudaraba", "moudarabah"), "13"),
        (("ijarah", "ijara", "location", "leasing", "crédit-bail"), "9"),
        (("salam",), "10"),
        (("istisna",), "11"),
        (("musharakah", "moucharaka", "partenariat", "associé"), "12"),
        (("sukuk", "sakk", "soukouk", "obligation", "obligations", "bond"), "17"),
        (("qard", "prêt", "emprunt"), "19"),
        (("action", "actions", "bourse", "shares", "stock"), "21"),
        (("assurance", "takaful", "insurance"), "26"),
        (("tawarruq", "tawaruq", "monetization", "monétisation", "liquidité", "trésorerie"), "30"),
        (("arbitrage", "arbitration"), "32"),
        (("waqf", "habous"), "33"),
        (("zakat", "zakah", "zakaat"), "35"),
        (("financer", "financement", "acheter", "achat", "équipement", "matériel", "véhicule", "immobilier"), "8"),
    )
    return {standard for terms, standard in mappings if any(term in normalized for term in terms)}

def retrieve(database_url: str, query_vector: np.ndarray, question: str) -> list[dict[str, Any]]:
    candidate_limit = max(TOP_K * 3, 15)
    sql = """
        select chunk_id, standard_number, standard_title, section,
               page_start, page_end, content,
               1 - (embedding <=> %s::vector) as similarity
        from public.rag_chunks
        order by embedding <=> %s::vector
        limit %s
    """
    vector = vector_literal(query_vector)
    with psycopg.connect(database_url, connect_timeout=30) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (vector, vector, candidate_limit))
            columns = [column.name for column in cursor.description]
            candidates = [dict(zip(columns, row)) for row in cursor.fetchall()]

    targets = preferred_standards(question)
    if not targets:
        return candidates[:TOP_K]

    existing_standards = {str(item.get("standard_number", "")).strip() for item in candidates}
    missing_targets = targets - existing_standards
    if missing_targets:
        targeted_sql = sql.replace(
            "order by embedding <=> %s::vector\nlimit %s",
            "where standard_number::text = %s\norder by embedding <=> %s::vector\nlimit %s",
        )
        with psycopg.connect(database_url, connect_timeout=30) as connection:
            with connection.cursor() as cursor:
                for target in sorted(missing_targets):
                    cursor.execute(targeted_sql, (vector, target, vector, TOP_K))
                    columns = [column.name for column in cursor.description]
                    candidates.extend(dict(zip(columns, row)) for row in cursor.fetchall())

    selected: list[dict[str, Any]] = []
    selected_ids: set[Any] = set()
    for target in targets:
        for item in candidates:
            standard = str(item.get("standard_number", "")).strip()
            item_id = item.get("chunk_id")
            if standard == target and item_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item_id)
                break

    for item in candidates:
        if item.get("chunk_id") not in selected_ids:
            selected.append(item)
        if len(selected) == TOP_K:
            break
    return selected

def format_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for index, item in enumerate(results, 1):
        content = (item.get("content", "") or "")[:MAX_CHUNK_CHARS]
        blocks.append(
            f"[SOURCE {index}]\n"
            f"AAOIFI Standard: {item.get('standard_number','?')}\n"
            f"Title: {item.get('standard_title','?')}\n"
            f"Pages: {item.get('page_start','?')}-{item.get('page_end','?')}\n"
            f"Content: {content}"
        )
    return "\n\n".join(blocks)

def repair_answer_citations(
    client: OpenAI, answer: str, results: list[dict[str, Any]], model_name: str,
) -> str:
    allowed = ", ".join(f"AAOIFI Standard {item.get('standard_number','?')}" for item in results)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a citation editor for a grounded AAOIFI answer. Return only the final answer.",
            },
            {
                "role": "user",
                "content": (
                    f"Allowed sources are only: {allowed}.\n\n"
                    "Use this exact citation format after each supported paragraph: "
                    "[Source : AAOIFI Standard X]. If a claim is not supported, replace it with: "
                    "The retrieved AAOIFI excerpts are insufficient to answer this question.\n\n"
                    f"Draft answer:\n{answer}"
                ),
            },
        ],
        temperature=0,
        max_tokens=MAX_GENERATION_TOKENS,
    )
    repaired = response.choices[0].message.content or ""
    return normalize_citations(clean_answer(repaired))

def create_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in .env")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=45.0, max_retries=0)

RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)\s*(ms|s)", re.IGNORECASE)

def suggested_wait_seconds(error_text: str, fallback: float) -> float:
    match = RETRY_AFTER_RE.search(error_text)
    if not match:
        return fallback
    value = float(match.group(1))
    if match.group(2).lower() == "ms":
        value /= 1000
    return value + 1.0  

def build_retrieval_query(question: str, recent_messages: list[dict[str, str]] | None) -> str:
    """Extracts domain intent cleanly without pollution from meta instructions."""
    if not recent_messages:
        return question
    
    user_turns = [
        m["content"] for m in recent_messages 
        if m["role"] == "user" and not any(term in m["content"].lower() for term in META_FORMATTING_TERMS)
    ]
    
    if not user_turns:
        return question
    
    return f"{user_turns[-1]} {question}".strip()

def summarize_conversation(
    recent_messages: list[dict[str, str]], 
    model_name: str = GROQ_MODEL
) -> str:
    """Summarizes recent conversation turns to keep context manageable."""
    if not recent_messages:
        return ""
    
    formatted_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in recent_messages
    )
    
    client = create_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Provide a concise 2-3 sentence summary of the key topics discussed in this conversation.",
                },
                {"role": "user", "content": formatted_history},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return formatted_history[-1000:]

def generate_answer(
    question: str,
    results: list[dict[str, Any]],
    model_name: str,
    persona_name: str,
    conversation_summary: str | None = None,
    recent_messages: list[dict[str, str]] | None = None,
    cross_conversation_context: str | None = None,
) -> str:
    persona_prompt = PERSONA_PROMPTS.get(persona_name, "")
    persona_instructions = (
        f"Active persona: {persona_name}. Apply this style in the final answer:\n"
        f"{persona_prompt}\n"
    )
    safe_question = redact_sensitive_data(question)

    cross_conv_block = (
        f"Cross-conversation history:\n{cross_conversation_context}\n\n"
        if cross_conversation_context else ""
    )
    
    summary_block = (
        f"Current session summary:\n{conversation_summary}\n\n"
        if conversation_summary else ""
    )

    prompt = (
        f"{persona_instructions}\n\n"
        f"{cross_conv_block}"
        f"{summary_block}"
        f"Question:\n{safe_question}\n\n"
        "Retrieved AAOIFI context:\n"
        f"{format_context(results)}\n\n"
        "Return only the final grounded answer. Finish every sentence. "
        "Use exact citation format: [Source : AAOIFI Standard X]."
    )

    conversation_turns = [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in (recent_messages or [])
    ]

    client = create_client()
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *conversation_turns,
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_GENERATION_TOKENS,
            )
            choice = response.choices[0]
            answer = choice.message.content
            if not answer or not answer.strip():
                raise RuntimeError("Groq returned an empty answer")
                
            answer = normalize_citations(clean_answer(answer))
            if not valid_citations(answer, results):
                answer = repair_answer_citations(client, answer, results, model_name)
            if not valid_citations(answer, results):
                return fallback_answer(results)
            return answer
        except Exception as error:
            last_error = error
            text = str(error).lower()
            temporary = any(marker in text for marker in ("429", "rate", "timeout", "connection", "temporar"))
            if not temporary or attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Groq request failed: {error}") from error
            wait = suggested_wait_seconds(text, fallback=5.0 * (attempt + 1))
            time.sleep(wait)
    raise RuntimeError(f"Groq request failed after retries: {last_error}")

def answer_question(
    question: str,
    database_url: str,
    model_name: str = GROQ_MODEL,
    persona_name: str = "general_public",
    conversation_summary: str | None = None,
    recent_messages: list[dict[str, str]] | None = None,
    cross_conversation_context: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:

    question = redact_sensitive_data(question.strip())
    if not question:
        raise ValueError("The question cannot be empty.")
        
    intent = classify_intent(question)
    client = create_client()
    
    # 1. MEMORY INTENT
    if intent == "MEMORY":
        if not cross_conversation_context and not conversation_summary:
            return "I don't have any recorded memory or context from previous conversations yet.", []
        
        memory_prompt = (
            "The user is asking about their past conversations or what you remember about them.\n\n"
            f"Cross-conversation Context:\n{cross_conversation_context or 'None'}\n\n"
            f"Current Session Summary:\n{conversation_summary or 'None'}\n\n"
            f"User Question: {question}\n\n"
            "Respond helpfully and naturally to the user based ONLY on the context provided above."
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Nourfi, an AI financial assistant. Answer memory and chat history queries based on the provided cross-conversation context."},
                {"role": "user", "content": memory_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip(), []

    # 2. CHITCHAT INTENT
    if intent == "CHITCHAT":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        return (response.choices[0].message.content or "").strip(), []

    # 3. PLATFORM INTENT
    if intent == "PLATFORM":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are Nourfi assistant. Answer user platform & chat usage questions helpfully and politely."},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return (response.choices[0].message.content or "").strip(), []

    # 4. FATWA INTENT
    if intent == "FATWA":
        return (
            "I cannot issue a fatwa or provide a definitive religious ruling. "
            "I can only summarize retrieved AAOIFI material for educational purposes.",
            [],
        )

    # 5. AAOIFI DOMAIN & META-FORMATTING INTENT
    tokenizer, encoder, device = load_encoder()

    retrieval_query = build_retrieval_query(question, recent_messages)
    query_vector = embed_query(retrieval_query, tokenizer, encoder, device)
    results = retrieve(database_url, query_vector, retrieval_query)

    if not results:
        return ("The retrieved AAOIFI excerpts are insufficient to answer this question.", [])
        
    answer = generate_answer(
        question,
        results,
        model_name,
        persona_name,
        conversation_summary,
        recent_messages,
        cross_conversation_context,
    )
    return answer, results