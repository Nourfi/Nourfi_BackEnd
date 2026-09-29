import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aaoifi_rag_chunks_15_standards_500.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_aaoifi_rag_chunks_15_standards_500.json"
)


EXPECTED_STANDARDS = [
    8,
    9,
    10,
    11,
    12,
    13,
    17,
    19,
    25,
    26,
    30,
    31,
    39,
    40,
    46,
]

MAX_TOKENS = 500
MAX_CONTENT_TOKENS = 425
MAX_OVERLAP_TOKENS = 75
MIN_CONTENT_TOKENS = 120


def create_result(
    name,
    passed,
    details=None,
    errors=None,
):
    return {
        "name": name,
        "passed": passed,
        "details": details or {},
        "errors": errors or [],
    }


def extract_standard_from_source_id(source_id):
    match = re.match(
        r"^SS(\d+)_",
        str(source_id or ""),
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def load_tokenizer():
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(
            "cl100k_base"
        )

    except ImportError as exc:
        raise RuntimeError(
            "Install tiktoken with: "
            "pip install tiktoken"
        ) from exc

    return encoding


def run_tests():
    results = []

    if not INPUT_FILE.exists():
        results.append(
            create_result(
                "input_file_exists",
                False,
                errors=[
                    f"File not found: {INPUT_FILE}"
                ],
            )
        )

        return results

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    metadata = data.get(
        "metadata",
        {},
    )

    chunks = data.get(
        "chunks",
        [],
    )

    actual_standards = sorted(
        {
            chunk.get("standard_number")
            for chunk in chunks
            if chunk.get("standard_number")
            is not None
        }
    )

    metadata_standards = sorted(
        metadata.get(
            "standard_numbers",
            [],
        )
    )

    results.append(
        create_result(
            "input_file_exists",
            True,
            details={
                "path": str(INPUT_FILE),
            },
        )
    )

    results.append(
        create_result(
            "chunks_array_exists",
            isinstance(chunks, list)
            and bool(chunks),
            details={
                "chunk_count": len(chunks),
            },
            errors=(
                []
                if isinstance(chunks, list)
                and chunks
                else ["The chunks array is empty."]
            ),
        )
    )

    results.append(
        create_result(
            "metadata_has_expected_standards",
            metadata_standards
            == EXPECTED_STANDARDS,
            details={
                "expected": EXPECTED_STANDARDS,
                "actual": metadata_standards,
            },
            errors=(
                []
                if metadata_standards
                == EXPECTED_STANDARDS
                else [
                    "Metadata does not contain exactly "
                    "the expected MVP standards."
                ]
            ),
        )
    )

    results.append(
        create_result(
            "chunks_have_expected_standards",
            actual_standards
            == EXPECTED_STANDARDS,
            details={
                "expected": EXPECTED_STANDARDS,
                "actual": actual_standards,
            },
            errors=(
                []
                if actual_standards
                == EXPECTED_STANDARDS
                else [
                    "Chunks do not contain exactly "
                    "the expected MVP standards."
                ]
            ),
        )
    )

    metadata_chunk_count = metadata.get(
        "num_chunks"
    )

    results.append(
        create_result(
            "metadata_chunk_count_matches",
            metadata_chunk_count
            == len(chunks),
            details={
                "metadata_count": metadata_chunk_count,
                "actual_count": len(chunks),
            },
            errors=(
                []
                if metadata_chunk_count
                == len(chunks)
                else [
                    "Metadata chunk count does not "
                    "match the chunks array."
                ]
            ),
        )
    )

    required_fields = [
        "chunk_id",
        "standard_number",
        "text",
        "token_count",
        "content_token_count",
        "overlap_token_count",
    ]

    missing_fields = []

    for index, chunk in enumerate(chunks):
        for field in required_fields:
            if field not in chunk:
                missing_fields.append(
                    {
                        "index": index,
                        "chunk_id": chunk.get(
                            "chunk_id"
                        ),
                        "missing_field": field,
                    }
                )

    results.append(
        create_result(
            "required_chunk_fields_exist",
            not missing_fields,
            details={
                "required_fields": required_fields,
                "missing_count": len(
                    missing_fields
                ),
            },
            errors=missing_fields[:25],
        )
    )

    duplicate_chunk_ids = []
    seen_chunk_ids = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            duplicate_chunk_ids.append(chunk_id)

        seen_chunk_ids.add(chunk_id)

    results.append(
        create_result(
            "chunk_ids_are_unique",
            not duplicate_chunk_ids,
            details={
                "duplicate_count": len(
                    duplicate_chunk_ids
                ),
            },
            errors=duplicate_chunk_ids[:25],
        )
    )

    try:
        encoding = load_tokenizer()
        tokenizer_error = None
    except RuntimeError as exc:
        encoding = None
        tokenizer_error = str(exc)

    if tokenizer_error:
        results.append(
            create_result(
                "tokenizer_available",
                False,
                errors=[tokenizer_error],
            )
        )

        return results

    results.append(
        create_result(
            "tokenizer_available",
            True,
            details={
                "tokenizer": "cl100k_base",
            },
        )
    )

    token_errors = []
    maximum_token_count = 0
    maximum_content_count = 0
    maximum_overlap_count = 0
    minimum_content_count = None

    for chunk in chunks:
        chunk_id = chunk.get(
            "chunk_id"
        )
        text = str(
            chunk.get(
                "text",
                "",
            )
        )

        actual_token_count = len(
            encoding.encode(text)
        )

        stored_token_count = chunk.get(
            "token_count"
        )

        content_token_count = chunk.get(
            "content_token_count",
            0,
        )

        overlap_token_count = chunk.get(
            "overlap_token_count",
            0,
        )

        maximum_token_count = max(
            maximum_token_count,
            actual_token_count,
        )

        maximum_content_count = max(
            maximum_content_count,
            content_token_count,
        )

        maximum_overlap_count = max(
            maximum_overlap_count,
            overlap_token_count,
        )

        if minimum_content_count is None:
            minimum_content_count = (
                content_token_count
            )
        else:
            minimum_content_count = min(
                minimum_content_count,
                content_token_count,
            )

        if actual_token_count != stored_token_count:
            token_errors.append(
                {
                    "chunk_id": chunk_id,
                    "type": "stored_count_mismatch",
                    "stored": stored_token_count,
                    "actual": actual_token_count,
                }
            )

        if actual_token_count > MAX_TOKENS:
            token_errors.append(
                {
                    "chunk_id": chunk_id,
                    "type": "maximum_tokens_exceeded",
                    "actual": actual_token_count,
                    "maximum": MAX_TOKENS,
                }
            )

        if content_token_count > MAX_CONTENT_TOKENS:
            token_errors.append(
                {
                    "chunk_id": chunk_id,
                    "type": "content_tokens_exceeded",
                    "actual": content_token_count,
                    "maximum": MAX_CONTENT_TOKENS,
                }
            )

        if overlap_token_count > MAX_OVERLAP_TOKENS:
            token_errors.append(
                {
                    "chunk_id": chunk_id,
                    "type": "overlap_tokens_exceeded",
                    "actual": overlap_token_count,
                    "maximum": MAX_OVERLAP_TOKENS,
                }
            )

    results.append(
        create_result(
            "token_counts_are_valid",
            not token_errors,
            details={
                "maximum_token_count": maximum_token_count,
                "maximum_content_count": maximum_content_count,
                "maximum_overlap_count": maximum_overlap_count,
                "minimum_content_count": minimum_content_count,
            },
            errors=token_errors[:50],
        )
    )

    first_chunk_errors = []
    seen_standards = set()

    for chunk in chunks:
        standard_number = chunk.get(
            "standard_number"
        )
        overlap_count = chunk.get(
            "overlap_token_count",
            0,
        )

        if standard_number not in seen_standards:
            seen_standards.add(standard_number)

            if overlap_count != 0:
                first_chunk_errors.append(
                    {
                        "chunk_id": chunk.get(
                            "chunk_id"
                        ),
                        "standard_number": standard_number,
                        "overlap_token_count": overlap_count,
                    }
                )

    results.append(
        create_result(
            "first_chunk_of_each_standard_has_no_overlap",
            not first_chunk_errors,
            details={
                "standards_checked": len(
                    seen_standards
                ),
            },
            errors=first_chunk_errors,
        )
    )

    boundary_errors = []

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        previous_standard = previous.get(
            "standard_number"
        )
        current_standard = current.get(
            "standard_number"
        )

        if (
            previous_standard
            != current_standard
            and current.get(
                "overlap_token_count",
                0,
            )
            != 0
        ):
            boundary_errors.append(
                {
                    "previous_chunk": previous.get(
                        "chunk_id"
                    ),
                    "current_chunk": current.get(
                        "chunk_id"
                    ),
                    "previous_standard": previous_standard,
                    "current_standard": current_standard,
                    "overlap_token_count": current.get(
                        "overlap_token_count",
                        0,
                    ),
                }
            )

    results.append(
        create_result(
            "overlap_does_not_cross_standards",
            not boundary_errors,
            errors=boundary_errors[:50],
        )
    )

    cross_standard_source_ids = []

    for chunk in chunks:
        chunk_standard = chunk.get(
            "standard_number"
        )

        for source_id in chunk.get(
            "source_chunk_ids",
            [],
        ):
            source_standard = (
                extract_standard_from_source_id(
                    source_id
                )
            )

            if (
                source_standard is not None
                and source_standard
                != chunk_standard
            ):
                cross_standard_source_ids.append(
                    {
                        "chunk_id": chunk.get(
                            "chunk_id"
                        ),
                        "chunk_standard": chunk_standard,
                        "source_id": source_id,
                        "source_standard": source_standard,
                    }
                )

    results.append(
        create_result(
            "source_ids_match_chunk_standards",
            not cross_standard_source_ids,
            details={
                "cross_standard_mapping_count": len(
                    cross_standard_source_ids
                ),
            },
            errors=cross_standard_source_ids[:50],
        )
    )

    small_chunks = []

    for chunk in chunks:
        content_count = chunk.get(
            "content_token_count",
            0,
        )

        if content_count < MIN_CONTENT_TOKENS:
            small_chunks.append(
                {
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "standard_number": chunk.get(
                        "standard_number"
                    ),
                    "content_token_count": content_count,
                }
            )

    results.append(
        create_result(
            "small_chunk_threshold",
            not small_chunks,
            details={
                "minimum_expected_content_tokens": (
                    MIN_CONTENT_TOKENS
                ),
                "small_chunk_count": len(
                    small_chunks
                ),
            },
            errors=small_chunks[:50],
        )
    )

    boundary_flags = [
        {
            "chunk_id": chunk.get(
                "chunk_id"
            ),
            "standard_number": chunk.get(
                "standard_number"
            ),
            "source_chunk_ids": chunk.get(
                "source_chunk_ids",
                [],
            ),
        }
        for chunk in chunks
        if chunk.get(
            "embedded_boundary_split",
            False,
        )
    ]

    results.append(
        create_result(
            "boundary_split_flags_reported",
            True,
            details={
                "flagged_chunk_count": len(
                    boundary_flags
                ),
                "examples": boundary_flags[:25],
            },
        )
    )

    return results


def main():
    results = run_tests()

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = len(results) - passed_count

    report = {
        "test_suite": (
            "AAOIFI RAG 15-standard chunk validation"
        ),
        "input_file": str(INPUT_FILE),
        "report_file": str(REPORT_FILE),
        "expected_standards": EXPECTED_STANDARDS,
        "test_count": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "status": (
            "passed"
            if failed_count == 0
            else "failed"
        ),
        "tests": results,
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("AAOIFI RAG CHUNK TESTS")
    print("=" * 70)
    print(f"[*] Input : {INPUT_FILE}")
    print(f"[*] Report: {REPORT_FILE}")
    print()
    print(f"[✓] Passed: {passed_count}")
    print(f"[✗] Failed: {failed_count}")
    print(f"[*] Status: {report['status']}")

    if failed_count:
        print(
            "\n[!] Some tests failed. "
            "Review the generated JSON report."
        )
        sys.exit(1)

    print(
        "\n[✓] All validation tests passed."
    )


if __name__ == "__main__":
    main()