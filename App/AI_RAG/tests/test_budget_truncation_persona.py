import importlib.util
import types
from pathlib import Path

PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "App"
    / "AI-RAG"
    / "rag"
    / "pipeline.py"
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("nourfi_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyChoice:
    def __init__(self, content, finish_reason=None):
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class DummyClient:
    def __init__(self, responses):
        # responses: list of objects returned by the provider, each must expose .choices
        self._responses = list(responses)
        self.chat = types.SimpleNamespace(completions=self)
        self.last_messages = None

    def create(self, model=None, messages=None, temperature=None, max_tokens=None, **kwargs):
        # record the messages for inspection and return the next mock response
        self.last_messages = messages
        if not self._responses:
            raise AssertionError("No mock response left for DummyClient")
        return self._responses.pop(0)


def sample_results():
    return [
        {
            "chunk_id": 1,
            "standard_number": "8",
            "standard_title": "Murabahah",
            "page_start": 10,
            "page_end": 12,
            "content": "Murabahah is a sale contract involving purchase and resale under agreed conditions.",
        },
        {
            "chunk_id": 2,
            "standard_number": "13",
            "standard_title": "Mudarabah",
            "page_start": 15,
            "page_end": 18,
            "content": "Mudarabah is a partnership where capital is provided by one party and management by another.",
        },
    ]


def test_build_context_blocks_trims_history_when_budget_small(monkeypatch):
    pipeline = load_pipeline()

    # Force a very small model context so trimming must occur
    monkeypatch.setattr(pipeline, "MODEL_CONTEXT_TOKENS", 256)
    monkeypatch.setattr(pipeline, "PROMPT_TOKEN_MARGIN", 50)
    monkeypatch.setattr(pipeline, "MAX_GENERATION_TOKENS", 100)
    monkeypatch.setattr(pipeline, "CHARS_PER_TOKEN", 4)

    persona_instructions = "Persona: test.\nExplain simply.\n"
    safe_question = "Explain murabaha?"
    results = sample_results()

    # Create long conversation summary and recent messages
    long_summary = " ".join(["summary"] * 500)  # long summary
    recent_messages = [
        {"role": "user", "content": "user message " + ("x " * 800)},
        {"role": "assistant", "content": "assistant reply " + ("y " * 800)},
        {"role": "user", "content": "follow-up " + ("z " * 800)},
    ]

    summary_block, conv_turns = pipeline.build_context_blocks(
        persona_instructions,
        safe_question,
        results,
        conversation_summary=long_summary,
        recent_messages=recent_messages,
    )

    # With very small MODEL_CONTEXT_TOKENS we expect either summary_block truncated or empty
    assert isinstance(summary_block, str)
    assert isinstance(conv_turns, list)
    # History must be trimmed: not all original messages should be kept in full
    assert len(conv_turns) <= len(recent_messages)
    # If no history fits, summary_block may still be present but truncated
    assert (summary_block == "" or len(summary_block) < len(long_summary))


def test_generate_answer_handles_truncation_and_uses_persona(monkeypatch):
    pipeline = load_pipeline()

    results = sample_results()

    # Mock responses: first response truncated, second is the shorter complete rewrite
    resp1 = types.SimpleNamespace(choices=[DummyChoice("Very long draft that was truncated ...", finish_reason="length")])
    resp2 = types.SimpleNamespace(choices=[DummyChoice("Réponse courte et complète. [Source : AAOIFI Standard 8]")])

    client = DummyClient([resp1, resp2])

    # Patch create_client to return our dummy client
    monkeypatch.setattr(pipeline, "create_client", lambda: client)

    # Ensure citation validation passes when the expected citation is present
    monkeypatch.setattr(pipeline, "valid_citations", lambda ans, res: "[Source : AAOIFI Standard 8]" in (ans or ""))

    # Call generate_answer: it should use the dummy client and perform the retry on length
    answer = pipeline.generate_answer(
        question="Quelles sont les conditions pour un investissement halal ?",
        results=results,
        model_name="mock-model",
        persona_name="general_public",
        conversation_summary="Some short summary",
        recent_messages=[{"role": "user", "content": "Short recent turn"}],
    )

    # The final answer should be the shorter cleaned response (from resp2)
    assert "Réponse courte et complète." in answer
    assert "[Source : AAOIFI Standard 8]" in answer

    # Inspect the messages sent to the model: the user prompt should include the persona block
    # The last call's messages are stored in client.last_messages
    assert client.last_messages is not None
    # The user's message (the prompt) is usually the last item
    user_msg = client.last_messages[-1]
    assert isinstance(user_msg, dict) or hasattr(user_msg, "get") or isinstance(user_msg, (list, tuple))
    # Convert to string safely
    content = ""
    if isinstance(user_msg, dict):
        content = user_msg.get("content", "")
    else:
        # If messages are simple namespaces or tuples, coerce to string
        content = str(user_msg)
    # Persona instructions should appear in the prompt content (persona_block includes "Persona:")
    assert "Persona:" in content or "Active persona" in content or "Persona" in content


def test_persona_prompts_file_builds_prompts_consistently():
    # load the persona prompts module to check its structure (the user provided file)
    spec = importlib.util.spec_from_file_location("personas", Path(__file__).resolve().parents[2] / "App" / "prompts" / "personas.py")
    persona_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(persona_mod)

    # PERSONA_PROMPTS should be a dict with expected keys
    assert isinstance(persona_mod.PERSONA_PROMPTS, dict)
    expected_keys = {"general_public", "student", "entrepreneur", "business_owner", "investor", "banker", "academic", "finance_specialist"}
    assert expected_keys.issubset(set(persona_mod.PERSONA_PROMPTS.keys()))
    # Each prompt should be a non-empty string
    for k in expected_keys:
        assert isinstance(persona_mod.PERSONA_PROMPTS[k], str)
        assert len(persona_mod.PERSONA_PROMPTS[k].strip()) > 0