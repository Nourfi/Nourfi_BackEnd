# chat_service.py — Conversation Memory Module (M4-02)
# Sol R Us (NOURFI) — Owner: Member 4 (AI/Front)

import json
from typing import Callable, List, Dict, Any

from App.prompts.base import PROMPT_BASE
from App.prompts.personas import PERSONA_PROMPTS

SUMMARY_FREQUENCY = 10


def get_recent_messages(
    conversation_history: List[Dict[str, Any]], limit: int = 10
) -> List[Dict[str, Any]]:
    return conversation_history[-limit:]


def generate_summary(
    old_summary: str, recent_messages: List[Dict[str, Any]], call_claude_fn: Callable
) -> str:
    formatted_messages = "\n".join(
        [f"{m['role']}: {m['content']}" for m in recent_messages]
    )
    previous = (
        f"Previous summary: {old_summary}"
        if old_summary
        else "No previous summary."
    )

    summary_prompt = (
        f"Summarize this conversation in 5 lines max:\n"
        f"topics/products discussed, objections handled, figures requested,\n"
        f"factual context given by the user. Stay neutral — never a recommendation\n"
        f"or \"the user should\".\n\n"
        f"{previous}\n\n"
        f"New messages:\n"
        f"{formatted_messages}"
    )

    return call_claude_fn([{"role": "user", "content": summary_prompt}], 300)


def build_context(
    summary: str,
    recent_messages: List[Dict[str, Any]],
    profile: Dict[str, Any],
    persona_key: str,
) -> List[Dict[str, Any]]:
    persona_prompt = PERSONA_PROMPTS.get(
        persona_key, PERSONA_PROMPTS["general_public"]
    )

    return [
        {"role": "system", "content": PROMPT_BASE},
        {"role": "system", "content": persona_prompt},
        {"role": "system", "content": f"User profile: {json.dumps(profile)}"},
        {
            "role": "system",
            "content": f"Conversation summary: {summary}"
            if summary
            else "No summary yet.",
        },
        *recent_messages,
    ]


def process_message(
    conversation: Dict[str, Any],
    profile: Dict[str, Any],
    persona_key: str,
    call_claude_fn: Callable,
) -> Dict[str, Any]:
    """
    Process an incoming message by checking if a summary update is needed,
    building context, and invoking the LLM function wrapper.
    """
    messages = conversation.get("messages", [])
    recent_messages = get_recent_messages(messages, 10)

    summary = conversation.get("summary")
    should_update_summary = (
        len(messages) > 0 and len(messages) % SUMMARY_FREQUENCY == 0
    )

    if should_update_summary:
        summary = generate_summary(summary, recent_messages, call_claude_fn)

    # Uses the OLD summary (conversation.get("summary")) for the main call to avoid redundancy
    context_for_main_call = build_context(
        conversation.get("summary"), recent_messages, profile, persona_key
    )
    ai_response = call_claude_fn(context_for_main_call, 1000)

    return {"aiResponse": ai_response, "updatedSummary": summary}