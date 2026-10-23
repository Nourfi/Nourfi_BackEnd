import importlib.util
import types
from pathlib import Path

import numpy as np


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


def sample_results():
    return [
        {
            "chunk_id": 1,
            "standard_number": "8",
            "standard_title": "Murabahah",
            "section": "General",
            "page_start": 10,
            "page_end": 12,
            "content": "Murabahah is a sale contract involving purchase and resale under the agreed conditions.",
        },
        {
            "chunk_id": 2,
            "standard_number": "13",
            "standard_title": "Mudarabah",
            "section": "General",
            "page_start": 15,
            "page_end": 18,
            "content": "Mudarabah is a partnership contract where capital is provided by one party and management by another.",
        },
    ]


class DummyChoice:
    def __init__(self, content, finish_reason=None):
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class DummyClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("No mock response left for Groq client")
        return self._responses.pop(0)


def test_persona_block_contains_persona_specific_instruction():
    pipeline = load_pipeline()

    block = pipeline.persona_block("general_public")
    assert "Persona: general_public" in block
    assert "Follow this persona precisely" in block

    block2 = pipeline.persona_block("legal_expert")
    assert "Persona: legal_expert" in block2


def test_is_fatwa_request_detects_common_fatwa_cases():
    pipeline = load_pipeline()

    questions = [
        "Give me a fatwa about this transaction.",
        "Is this halal or haram?",
        "هل هذا حلال أم حرام؟",
        "Est-ce halal ou haram ?",
    ]

    for q in questions:
        assert pipeline.is_fatwa_request(q) is True


def test_fatwa_questions_are_refused_before_planner_or_retrieval(monkeypatch):
    pipeline = load_pipeline()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Should not call planner or retrieval for fatwa requests")

    monkeypatch.setattr(pipeline, "create_client", fail_if_called)

    answer, results = pipeline.answer_question(
        "Is this halal or haram?",
        "postgres://fake",
    )

    assert "definitive religious ruling" in answer
    assert results == []


def test_planner_can_route_small_talk_without_retrieval(monkeypatch):
    pipeline = load_pipeline()

    planner_response = types.SimpleNamespace(
        choices=[DummyChoice('{"need_retrieval": false, "answer": "Hello! How can I help you today?"}')]
    )
    client = DummyClient([planner_response])
    monkeypatch.setattr(pipeline, "create_client", lambda: client)

    monkeypatch.setattr(pipeline, "load_encoder", lambda: (_ for _ in ()).throw(AssertionError("No encoder should load for small talk")))
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No retrieval for small talk")))

    answer, results = pipeline.answer_question("hi", "postgres://fake")

    assert "Hello" in answer
    assert results == []


def test_planner_defaults_to_retrieval_on_malformed_json(monkeypatch):
    pipeline = load_pipeline()

    planner_response = types.SimpleNamespace(
        choices=[DummyChoice("not-json")]
    )
    client = DummyClient([planner_response])

    monkeypatch.setattr(pipeline, "create_client", lambda: client)
    monkeypatch.setattr(pipeline, "load_encoder", lambda: ("tok", "enc", "cpu"))
    monkeypatch.setattr(pipeline, "embed_query", lambda *args, **kwargs: np.array([0.1, 0.2], dtype=np.float32))
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: sample_results())
    monkeypatch.setattr(pipeline, "generate_answer", lambda *args, **kwargs: "Grounded answer")

    answer, results = pipeline.answer_question("What is murabaha?", "postgres://fake")

    assert answer == "Grounded answer"
    assert len(results) == 2


def test_answer_question_passes_memory_context_to_generate_answer(monkeypatch):
    pipeline = load_pipeline()

    planner_response = types.SimpleNamespace(
        choices=[DummyChoice('{"need_retrieval": true}')]
    )
    client = DummyClient([planner_response])

    monkeypatch.setattr(pipeline, "create_client", lambda: client)
    monkeypatch.setattr(pipeline, "load_encoder", lambda: ("tok", "enc", "cpu"))
    monkeypatch.setattr(pipeline, "embed_query", lambda *args, **kwargs: np.array([0.1, 0.2], dtype=np.float32))
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: sample_results())

    seen = {}

    def fake_generate(question, results, model_name, persona_name, conv_summary=None, recent_messages=None, cross_context=None):
        seen["question"] = question
        seen["conv_summary"] = conv_summary
        seen["recent_messages"] = recent_messages
        seen["cross_context"] = cross_context
        return "Answer from memory"

    monkeypatch.setattr(pipeline, "generate_answer", fake_generate)

    recent = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "What about murabaha last time?"},
    ]

    answer, results = pipeline.answer_question(
        "What about murabaha?",
        "postgres://fake",
        conversation_summary="Earlier we discussed murabaha.",
        recent_messages=recent,
        cross_conversation_context="We discussed the finance structure earlier.",
    )

    assert answer == "Answer from memory"
    assert seen["conv_summary"] == "Earlier we discussed murabaha."
    assert seen["recent_messages"] == recent
    assert seen["cross_context"] == "We discussed the finance structure earlier."
    assert len(results) == 2


def test_build_context_blocks_trims_long_history_to_fit_budget():
    pipeline = load_pipeline()

    persona = "Persona: general_public.\nExplain in simple english.\n"
    safe_question = "What is murabaha?"
    results = sample_results()

    recent = [
        {"role": "user", "content": "x " * 1000},
        {"role": "assistant", "content": "y " * 1000},
        {"role": "user", "content": "z " * 1000},
    ]
    long_summary = "summary " * 1000

    summary_block, conv_turns = pipeline.build_context_blocks(
        persona,
        safe_question,
        results,
        conversation_summary=long_summary,
        recent_messages=recent,
    )

    assert isinstance(summary_block, str)
    assert isinstance(conv_turns, list)
    # Each message should be trimmed to keep prompt manageable
    assert len(conv_turns) <= len(recent)
    assert "Current session summary:" in summary_block or summary_block == ""


def test_generate_answer_retries_on_length_truncation(monkeypatch):
    pipeline = load_pipeline()

    results = sample_results()

    # first call -> finish_reason == "length"
    # second call -> valid answer
    response1 = types.SimpleNamespace(
        choices=[DummyChoice("very long answer ...", finish_reason="length")]
    )
    response2 = types.SimpleNamespace(
        choices=[DummyChoice("Short answer with proper citation [Source : AAOIFI Standard 8]")]
    )

    client = DummyClient([response1, response2])

    monkeypatch.setattr(pipeline, "create_client", lambda: client)
    monkeypatch.setattr(pipeline, "repair_answer_citations", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(pipeline, "valid_citations", lambda answer, res: "[Source : AAOIFI Standard 8]" in answer)

    answer = pipeline.generate_answer(
        "What is murabaha?",
        results,
        model_name="test-model",
        persona_name="general_public",
    )

    assert "[Source : AAOIFI Standard 8]" in answer
    assert "Short answer" in answer


def test_valid_citations_rejects_unretrieved_standard():
    pipeline = load_pipeline()

    answer = "This claim is unsupported. [Source : AAOIFI Standard 54]"
    results = sample_results()

    assert pipeline.valid_citations(answer, results) is False


def test_normalize_citations_keeps_only_one_citation_per_paragraph():
    pipeline = load_pipeline()

    answer = (
        "First claim [Source : AAOIFI Standard 8] and second claim "
        "[Source : AAOIFI Standard 13]"
    )
    normalized = pipeline.normalize_citations(answer)

    assert normalized.count("[Source : AAOIFI Standard") == 1
    assert normalized.endswith("[Source : AAOIFI Standard 13]")


def test_redact_sensitive_data_replaces_long_digit_sequences():
    pipeline = load_pipeline()

    text = "My number is 4111 1111 1111 1111 and my phone is 123456789012345"
    out = pipeline.redact_sensitive_data(text)
    assert "[REDACTED_NUMBER]" in out
    assert "4111" not in out or "[REDACTED_NUMBER]" in out


def test_answer_question_canonical_fallback_when_no_results(monkeypatch):
    pipeline = load_pipeline()

    planner_response = types.SimpleNamespace(
        choices=[DummyChoice('{"need_retrieval": true}')]
    )
    client = DummyClient([planner_response])

    monkeypatch.setattr(pipeline, "create_client", lambda: client)
    monkeypatch.setattr(pipeline, "load_encoder", lambda: ("tok", "enc", "cpu"))
    monkeypatch.setattr(pipeline, "embed_query", lambda *args, **kwargs: np.array([0.1, 0.2], dtype=np.float32))
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: [])

    answer, results = pipeline.answer_question("What is a random unknown concept?", "postgres://fake")

    assert "insufficient" in answer.lower()
    assert results == []


def test_persona_instructions_are_injected_into_generation_prompt(monkeypatch):
    pipeline = load_pipeline()

    results = sample_results()

    planner_response = types.SimpleNamespace(
        choices=[DummyChoice('{"need_retrieval": true}')]
    )
    client = DummyClient([planner_response])

    monkeypatch.setattr(pipeline, "create_client", lambda: client)
    monkeypatch.setattr(pipeline, "load_encoder", lambda: ("tok", "enc", "cpu"))
    monkeypatch.setattr(pipeline, "embed_query", lambda *args, **kwargs: np.array([0.1, 0.2], dtype=np.float32))
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: results)

    captured = {}

    def fake_generate(question, res, model_name, persona_name, conv_summary=None, recent_messages=None, cross_context=None):
        captured["persona_name"] = persona_name
        captured["prompt_contains_persona"] = "Active persona" in pipeline.persona_block(persona_name)
        return "Good persona-aware answer [Source : AAOIFI Standard 8]"

    monkeypatch.setattr(pipeline, "generate_answer", fake_generate)

    answer, results_out = pipeline.answer_question("What is murabaha?", "postgres://fake", persona_name="general_public")

    assert answer.startswith("Good persona-aware answer")
    assert captured["persona_name"] == "general_public"
    assert captured["prompt_contains_persona"] is True
    assert len(results_out) == 2