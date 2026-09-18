"""TST-04: AAOIFI citation verification tests.

These tests are offline. They validate citation parsing against retrieved
standard metadata without calling Supabase, Hugging Face, or OpenRouter.
"""

import importlib.util
from pathlib import Path


PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rag"
    / "pipeline.py"
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("nourfi_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retrieved_results():
    return [
        {"standard_number": 24, "standard_title": "Sukuk"},
        {"standard_number": 8, "standard_title": "Murabahah"},
    ]


def test_accepts_exact_citations_from_retrieved_standards():
    pipeline = load_pipeline()

    answer = (
        "The retrieved material describes the relevant requirements. "
        "[Source : AAOIFI Standard 24]"
    )

    assert pipeline.valid_citations(answer, retrieved_results())


def test_rejects_citation_for_standard_not_retrieved():
    pipeline = load_pipeline()

    answer = "This claim is unsupported. [Source : AAOIFI Standard 54]"

    assert not pipeline.valid_citations(answer, retrieved_results())


def test_rejects_answer_without_citation():
    pipeline = load_pipeline()

    assert not pipeline.valid_citations(
        "The answer contains no source.",
        retrieved_results(),
    )


def test_rejects_noncanonical_citation_format():
    pipeline = load_pipeline()

    answer = "Claim [Source: AAOIFI Standard 24]"

    assert not pipeline.valid_citations(answer, retrieved_results())


def test_normalizes_duplicate_citations_in_one_paragraph():
    pipeline = load_pipeline()

    answer = (
        "First claim [Source : AAOIFI Standard 24] and second claim "
        "[Source : AAOIFI Standard 8]"
    )

    normalized = pipeline.normalize_citations(answer)

    assert normalized.count("[Source : AAOIFI Standard") == 1
    assert normalized.endswith("[Source : AAOIFI Standard 8]")
