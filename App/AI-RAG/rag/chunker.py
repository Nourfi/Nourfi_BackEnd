
"""
NourFi — RAG Chunker 

Usage:

    python App\AI-RAG\rag\chunker.py --corpus 15
    python App\AI-RAG\rag\chunker.py --corpus 54

Features:
- 425 content tokens.
- Up to 75 tokens of overlap.
- Maximum 500 tokens per final chunk.
- Hard boundaries between standards.
- Supports 15-standard and 54-standard corpora.
- Skips front matter and chunks without a standard number.
- Handles OCR-separated standard headings.
- Ignores inline standard references.
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


BASE_DIR = Path(__file__).resolve().parents[2]


CONTENT_TARGET = 425
OVERLAP_TOKENS = 75
FINAL_MAX_TOKENS = 500

SOURCE_OVERLAP_MAX_TOKENS = 200
SOURCE_OVERLAP_MIN_TOKENS = 12

MIN_MERGE_TOKENS = 120


MVP_STANDARD_NUMBERS = {
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
}


CORPUS_CONFIG = {
    "15": {
        "input": (
            BASE_DIR
            / "data"
            / "aaoifi_rag_corpus_15_standards.json"
        ),
        "output": (
            BASE_DIR
            / "data"
            / "processed"
            / "aaoifi_rag_chunks_15_standards_500.json"
        ),
        "standards": MVP_STANDARD_NUMBERS,
    },
    "54": {
        "input": (
            BASE_DIR
            / "data"
            / "aaoifi_rag_corpus_54_standards.json"
        ),
        "output": (
            BASE_DIR
            / "data"
            / "processed"
            / "aaoifi_rag_chunks_54_standards_500.json"
        ),
        "standards": None,
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate NourFi RAG chunks."
    )

    parser.add_argument(
        "--corpus",
        choices=("15", "54"),
        default="15",
        help=(
            "Select the corpus: 15 for the MVP corpus "
            "or 54 for the complete corpus."
        ),
    )

    return parser.parse_args()


def load_tokenizer():
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        print("[✓] tiktoken loaded: cl100k_base")

    except ImportError as exc:
        raise RuntimeError(
            "Install tiktoken with: pip install tiktoken"
        ) from exc

    def encode(text):
        return encoding.encode(text)

    def decode(token_ids):
        return encoding.decode(token_ids)

    def count(text):
        return len(encoding.encode(text))

    return encode, decode, count


def normalize_text(text):
    text = str(text or "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Preserve newlines for heading detection.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_source_chunk_id(source_chunk):
    return (
        source_chunk.get("chunk_id")
        or source_chunk.get("source_id")
        or source_chunk.get("id")
    )


def extract_standard_from_source_id(source_id):
    match = re.match(
        r"^SS(\d+)_",
        str(source_id or ""),
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def get_source_standard_number(source_chunk):
    value = source_chunk.get("standard_number")

    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    source_id = get_source_chunk_id(source_chunk)
    value = extract_standard_from_source_id(source_id)

    if value is not None:
        return value

    for source_id in source_chunk.get(
        "source_chunk_ids",
        [],
    ):
        value = extract_standard_from_source_id(source_id)

        if value is not None:
            return value

    return None


def detect_source_standards(source_chunks):
    standards = set()

    for source_chunk in source_chunks:
        number = get_source_standard_number(
            source_chunk
        )

        if number is not None:
            standards.add(number)

    return standards


def find_real_standard_headings(
    text,
    allowed_standard_numbers,
):
    """
    Detects genuine headings such as:

        Shari'ah Standard No. (31)

    and OCR-separated headings such as:

        Shari'ah Standard No.

        (31)

    Inline references are ignored because the heading must end after
    the standard number.
    """
    pattern = re.compile(
        r"(?im)"
        r"^[ \t]*"
        r"Shari[’'`]?ah"
        r"[ \t]+Standard[ \t]+No\.?"
        r"[ \t]*"
        r"(?:\n[ \t]*)?"
        r"\(\s*(\d+)\s*\)"
        r"[ \t]*$"
    )

    matches = []

    for match in pattern.finditer(text):
        number = int(match.group(1))

        if number in allowed_standard_numbers:
            matches.append(match)

    return matches


def split_embedded_standard_boundaries(
    source_chunks,
    allowed_standard_numbers,
):
    """
    Splits a source chunk whenever it contains a real standard heading.

    The text before the first heading keeps the source metadata's
    standard number. Every heading assigns its following text to the
    heading's standard number.
    """
    result = []

    for source_chunk in source_chunks:
        text = normalize_text(
            source_chunk.get("text", "")
        )

        if not text:
            continue

        declared_standard = (
            get_source_standard_number(source_chunk)
        )

        if declared_standard is None:
            continue

        matches = find_real_standard_headings(
            text,
            allowed_standard_numbers,
        )

        if not matches:
            if declared_standard in allowed_standard_numbers:
                item = dict(source_chunk)
                item["text"] = text
                item["standard_number"] = (
                    declared_standard
                )
                item["embedded_boundary_split"] = False
                result.append(item)

            continue

        boundaries = []

        for match in matches:
            number = int(match.group(1))

            if (
                not boundaries
                or match.start()
                != boundaries[-1][0]
            ):
                boundaries.append(
                    (
                        match.start(),
                        number,
                    )
                )

        # Preserve any prefix before the first heading.
        first_start = boundaries[0][0]

        if first_start > 0:
            prefix = text[:first_start].strip()

            if (
                prefix
                and declared_standard
                in allowed_standard_numbers
            ):
                item = dict(source_chunk)
                item["text"] = prefix
                item["standard_number"] = (
                    declared_standard
                )
                item["embedded_boundary_split"] = True
                item["embedded_part_index"] = 0
                result.append(item)

        for index, (start, number) in enumerate(
            boundaries
        ):
            end = (
                boundaries[index + 1][0]
                if index + 1 < len(boundaries)
                else len(text)
            )

            part_text = text[start:end].strip()

            if (
                not part_text
                or number not in allowed_standard_numbers
            ):
                continue

            item = dict(source_chunk)
            item["text"] = part_text
            item["standard_number"] = number
            item["embedded_boundary_split"] = True
            item["embedded_part_index"] = index + 1
            result.append(item)

    return result


def remove_adjacent_source_overlap(
    source_chunks,
    encode,
    decode,
):
    """
    Removes exact token overlap between adjacent pieces of the same
    standard. Final RAG overlap is added later.
    """
    cleaned = []
    previous_standard = None
    previous_tokens = []

    for source_chunk in source_chunks:
        item = dict(source_chunk)
        text = normalize_text(
            item.get("text", "")
        )

        if not text:
            continue

        tokens = encode(text)
        removed = 0

        same_standard = (
            previous_standard is not None
            and item.get("standard_number")
            == previous_standard
        )

        if (
            same_standard
            and previous_tokens
            and tokens
        ):
            maximum = min(
                SOURCE_OVERLAP_MAX_TOKENS,
                len(previous_tokens),
                len(tokens),
            )

            for size in range(
                maximum,
                SOURCE_OVERLAP_MIN_TOKENS - 1,
                -1,
            ):
                if (
                    previous_tokens[-size:]
                    == tokens[:size]
                ):
                    tokens = tokens[size:]
                    text = decode(tokens).strip()
                    removed = size
                    break

        if not text:
            previous_standard = item.get(
                "standard_number"
            )
            previous_tokens = []
            continue

        item["text"] = text
        item["token_count"] = len(tokens)
        item["char_count"] = len(text)
        item["source_overlap_removed_tokens"] = (
            removed
        )

        cleaned.append(item)

        previous_standard = item.get(
            "standard_number"
        )
        previous_tokens = tokens

    return cleaned


def split_by_token_window(
    text,
    max_tokens,
    encode,
    decode,
):
    tokens = encode(text)
    pieces = []

    for start in range(
        0,
        len(tokens),
        max_tokens,
    ):
        piece = decode(
            tokens[start:start + max_tokens]
        ).strip()

        if piece:
            pieces.append(piece)

    return pieces


def split_long_text(
    text,
    max_tokens,
    encode,
    decode,
    count,
):
    text = normalize_text(text)

    if not text:
        return []

    if count(text) <= max_tokens:
        return [text]

    paragraphs = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]

    if len(paragraphs) > 1:
        result = []

        for part in paragraphs:
            result.extend(
                split_long_text(
                    part,
                    max_tokens,
                    encode,
                    decode,
                    count,
                )
            )

        return result

    sentences = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?;:])\s+",
            text,
        )
        if part.strip()
    ]

    if len(sentences) > 1:
        result = []

        for part in sentences:
            result.extend(
                split_long_text(
                    part,
                    max_tokens,
                    encode,
                    decode,
                    count,
                )
            )

        return result

    clauses = [
        part.strip()
        for part in re.split(
            r"(?<=,)\s+",
            text,
        )
        if part.strip()
    ]

    if len(clauses) > 1:
        result = []

        for part in clauses:
            result.extend(
                split_long_text(
                    part,
                    max_tokens,
                    encode,
                    decode,
                    count,
                )
            )

        return result

    return split_by_token_window(
        text,
        max_tokens,
        encode,
        decode,
    )


def atomicize_source_chunks(
    source_chunks,
    encode,
    decode,
    count,
):
    result = []

    for source_chunk in source_chunks:
        pieces = split_long_text(
            source_chunk.get("text", ""),
            CONTENT_TARGET,
            encode,
            decode,
            count,
        )

        for index, text in enumerate(
            pieces,
            start=1,
        ):
            item = dict(source_chunk)
            item["text"] = text
            item["atomic_index"] = index
            item["token_count"] = count(text)
            item["char_count"] = len(text)
            result.append(item)

    return result


def pack_content_chunks(
    atomic_pieces,
    count,
):
    result = []
    buffer = []
    current_standard = None

    def flush():
        nonlocal buffer

        if not buffer:
            return

        text = "\n\n".join(
            item["text"]
            for item in buffer
        )

        token_count = count(text)

        if token_count > CONTENT_TARGET:
            raise RuntimeError(
                "Content chunk exceeds target: "
                f"{token_count} > {CONTENT_TARGET}"
            )

        source_ids = []

        for item in buffer:
            source_id = get_source_chunk_id(item)

            if (
                source_id
                and source_id not in source_ids
            ):
                source_ids.append(source_id)

        first = buffer[0]
        last = buffer[-1]

        result.append(
            {
                "content_index": len(result) + 1,
                "standard_number": first[
                    "standard_number"
                ],
                "standard_title": first.get(
                    "standard_title",
                    "",
                ),
                "section": first.get(
                    "section",
                    "body",
                ),
                "page_start": first.get(
                    "page_start"
                ),
                "page_end": last.get(
                    "page_end"
                ),
                "source_chunk_ids": source_ids,
                "source_id": first.get(
                    "source_id"
                ),
                "source_type": first.get(
                    "source_type",
                    "official_standard",
                ),
                "language": first.get(
                    "language",
                    "en",
                ),
                "copyright_restricted": first.get(
                    "copyright_restricted",
                    True,
                ),
                "embedded_boundary_split": any(
                    item.get(
                        "embedded_boundary_split",
                        False,
                    )
                    for item in buffer
                ),
                "text": text,
                "char_count": len(text),
                "content_token_count": token_count,
            }
        )

        buffer = []

    for piece in atomic_pieces:
        standard_number = piece[
            "standard_number"
        ]

        if (
            current_standard is not None
            and standard_number != current_standard
        ):
            flush()

        current_standard = standard_number

        candidate = "\n\n".join(
            [
                item["text"]
                for item in buffer
            ]
            + [piece["text"]]
        )

        if (
            buffer
            and count(candidate) > CONTENT_TARGET
        ):
            flush()

        buffer.append(piece)

        buffer_text = "\n\n".join(
            item["text"]
            for item in buffer
        )

        if count(buffer_text) >= CONTENT_TARGET:
            flush()

    flush()
    return result


def merge_small_chunks(
    chunks,
    count,
):
    """
    Merges small chunks where possible without crossing a standard
    boundary.
    """
    chunks = [dict(chunk) for chunk in chunks]
    changed = True

    while changed:
        changed = False

        for index, current in enumerate(chunks):
            if count(current["text"]) >= MIN_MERGE_TOKENS:
                continue

            previous = (
                chunks[index - 1]
                if index > 0
                else None
            )

            following = (
                chunks[index + 1]
                if index + 1 < len(chunks)
                else None
            )

            if previous is not None:
                candidate = (
                    f'{previous["text"]}\n\n'
                    f'{current["text"]}'
                )

                if (
                    previous["standard_number"]
                    == current["standard_number"]
                    and count(candidate)
                    <= CONTENT_TARGET
                ):
                    previous["text"] = candidate
                    previous["page_end"] = current.get(
                        "page_end"
                    )
                    previous["char_count"] = len(
                        candidate
                    )
                    previous[
                        "content_token_count"
                    ] = count(candidate)

                    for source_id in current.get(
                        "source_chunk_ids",
                        [],
                    ):
                        if source_id not in previous[
                            "source_chunk_ids"
                        ]:
                            previous[
                                "source_chunk_ids"
                            ].append(source_id)

                    chunks.pop(index)
                    changed = True
                    break

            if following is not None:
                candidate = (
                    f'{current["text"]}\n\n'
                    f'{following["text"]}'
                )

                if (
                    current["standard_number"]
                    == following["standard_number"]
                    and count(candidate)
                    <= CONTENT_TARGET
                ):
                    following["text"] = candidate
                    following["page_start"] = current.get(
                        "page_start"
                    )
                    following["char_count"] = len(
                        candidate
                    )
                    following[
                        "content_token_count"
                    ] = count(candidate)

                    current_ids = current.get(
                        "source_chunk_ids",
                        [],
                    )
                    following_ids = following.get(
                        "source_chunk_ids",
                        [],
                    )

                    following[
                        "source_chunk_ids"
                    ] = (
                        current_ids
                        + [
                            source_id
                            for source_id in following_ids
                            if source_id
                            not in current_ids
                        ]
                    )

                    chunks.pop(index)
                    changed = True
                    break

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk["content_index"] = index
        chunk["char_count"] = len(
            chunk["text"]
        )
        chunk["content_token_count"] = count(
            chunk["text"]
        )

    return chunks


def build_final_chunks(
    content_chunks,
    encode,
    decode,
    count,
):
    result = []
    previous_content = None

    for content in content_chunks:
        standard_number = content[
            "standard_number"
        ]
        content_tokens = encode(
            content["text"]
        )

        if (
            previous_content is not None
            and previous_content[
                "standard_number"
            ] == standard_number
        ):
            previous_tokens = encode(
                previous_content["text"]
            )
            overlap_tokens = previous_tokens[
                -OVERLAP_TOKENS:
            ]
        else:
            previous_tokens = []
            overlap_tokens = []

        separator_tokens = (
            encode("\n\n")
            if overlap_tokens
            else []
        )

        final_tokens = (
            overlap_tokens
            + separator_tokens
            + content_tokens
        )

        if len(final_tokens) > FINAL_MAX_TOKENS:
            available = max(
                0,
                FINAL_MAX_TOKENS
                - len(separator_tokens)
                - len(content_tokens),
            )

            overlap_tokens = (
                previous_tokens[-available:]
                if available
                else []
            )

            separator_tokens = (
                encode("\n\n")
                if overlap_tokens
                else []
            )

            final_tokens = (
                overlap_tokens
                + separator_tokens
                + content_tokens
            )

        final_text = decode(final_tokens)
        final_count = count(final_text)

        if final_count > FINAL_MAX_TOKENS:
            raise RuntimeError(
                f"Final chunk exceeds "
                f"{FINAL_MAX_TOKENS}: "
                f"{final_count} tokens"
            )

        chunk_id = (
            f"STD{standard_number:02d}_"
            f"C{content['content_index']:03d}"
        )

        result.append(
            {
                "chunk_id": chunk_id,
                "standard_number": standard_number,
                "standard_title": content[
                    "standard_title"
                ],
                "section": content["section"],
                "page_start": content[
                    "page_start"
                ],
                "page_end": content["page_end"],
                "source_chunk_ids": content[
                    "source_chunk_ids"
                ],
                "source_id": content[
                    "source_id"
                ],
                "source_type": content[
                    "source_type"
                ],
                "language": content[
                    "language"
                ],
                "copyright_restricted": content[
                    "copyright_restricted"
                ],
                "content_chunk_id": chunk_id,
                "content_token_count": len(
                    content_tokens
                ),
                "overlap_token_count": len(
                    overlap_tokens
                ),
                "has_overlap": bool(
                    overlap_tokens
                ),
                "embedded_boundary_split": content.get(
                    "embedded_boundary_split",
                    False,
                ),
                "char_count": len(final_text),
                "token_count": final_count,
                "text": final_text,
            }
        )

        previous_content = content

    return result


def validate_chunks(
    chunks,
    count,
    expected_standards,
):
    if not chunks:
        raise RuntimeError(
            "No chunks were generated."
        )

    for chunk in chunks:
        actual_count = count(
            chunk["text"]
        )

        if actual_count != chunk[
            "token_count"
        ]:
            raise RuntimeError(
                f"Token mismatch: "
                f"{chunk['chunk_id']}"
            )

        if actual_count > FINAL_MAX_TOKENS:
            raise RuntimeError(
                f"Chunk exceeds maximum: "
                f"{chunk['chunk_id']}"
            )

        if chunk[
            "content_token_count"
        ] > CONTENT_TARGET:
            raise RuntimeError(
                f"Content exceeds target: "
                f"{chunk['chunk_id']}"
            )

        if chunk[
            "overlap_token_count"
        ] > OVERLAP_TOKENS:
            raise RuntimeError(
                f"Overlap exceeds limit: "
                f"{chunk['chunk_id']}"
            )

        if chunk[
            "standard_number"
        ] not in expected_standards:
            raise RuntimeError(
                f"Unexpected standard in output: "
                f"{chunk['standard_number']}"
            )

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        if (
            previous["standard_number"]
            != current["standard_number"]
            and current["overlap_token_count"] != 0
        ):
            raise RuntimeError(
                "Overlap crosses a standard boundary: "
                f"{previous['chunk_id']} -> "
                f"{current['chunk_id']}"
            )


def get_standards_in_order(chunks):
    result = []

    for chunk in chunks:
        number = chunk.get(
            "standard_number"
        )

        if (
            number is not None
            and number not in result
        ):
            result.append(number)

    return result


def main():
    args = parse_arguments()
    config = CORPUS_CONFIG[args.corpus]

    input_path = config["input"]
    output_path = config["output"]
    configured_standards = config["standards"]

    print("=" * 70)
    print(
        f"NOURFI — RAG CHUNKER v9 "
        f"({args.corpus}-STANDARD MODE)"
    )
    print("=" * 70)
    print(f"[*] Input          : {input_path}")
    print(f"[*] Output         : {output_path}")
    print(f"[*] Content target : {CONTENT_TARGET}")
    print(f"[*] Final overlap  : {OVERLAP_TOKENS}")
    print(f"[*] Final maximum  : {FINAL_MAX_TOKENS}")
    print()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        corpus = json.load(file)

    raw_chunks = corpus.get(
        "chunks",
        [],
    )

    if not raw_chunks:
        raise ValueError(
            "Input corpus contains no chunks."
        )

    source_chunks = []
    skipped_chunks = []

    for source_chunk in raw_chunks:
        number = get_source_standard_number(
            source_chunk
        )

        # Front matter, appendices without IDs, and other metadata-only
        # chunks are ignored.
        if number is None:
            skipped_chunks.append(
                get_source_chunk_id(source_chunk)
            )
            continue

        source_chunk = dict(source_chunk)
        source_chunk["standard_number"] = number
        source_chunks.append(source_chunk)

    if configured_standards is None:
        allowed_standards = detect_source_standards(
            source_chunks
        )
    else:
        allowed_standards = set(
            configured_standards
        )

    if not allowed_standards:
        raise RuntimeError(
            "No standard numbers were detected."
        )

    source_chunks = [
        chunk
        for chunk in source_chunks
        if chunk["standard_number"]
        in allowed_standards
    ]

    print(
        f"[✓] Source chunks loaded: "
        f"{len(raw_chunks)}"
    )
    print(
        f"[✓] Non-standard chunks skipped: "
        f"{len(skipped_chunks)}"
    )

    if skipped_chunks:
        print(
            "    Examples: "
            f"{skipped_chunks[:5]}"
        )

    print(
        f"[✓] Standards selected: "
        f"{len(allowed_standards)}"
    )
    print(
        f"    Numbers: "
        f"{sorted(allowed_standards)}"
    )

    source_chunks = (
        split_embedded_standard_boundaries(
            source_chunks,
            allowed_standards,
        )
    )

    source_chunks = [
        chunk
        for chunk in source_chunks
        if chunk.get(
            "standard_number"
        ) in allowed_standards
    ]

    print(
        "[✓] Boundary-aware source pieces: "
        f"{len(source_chunks)}"
    )

    encode, decode, count = load_tokenizer()

    standards = get_standards_in_order(
        source_chunks
    )

    unexpected = set(standards) - allowed_standards

    if unexpected:
        raise RuntimeError(
            "Unexpected standards detected: "
            f"{sorted(unexpected)}"
        )

    all_final = []

    for standard_number in standards:
        standard_source = [
            chunk
            for chunk in source_chunks
            if chunk["standard_number"]
            == standard_number
        ]

        print(
            f"\n[*] Processing Standard "
            f"{standard_number}"
        )
        print(
            f"    Source pieces: "
            f"{len(standard_source)}"
        )

        cleaned_source = (
            remove_adjacent_source_overlap(
                standard_source,
                encode,
                decode,
            )
        )

        removed_tokens = sum(
            item.get(
                "source_overlap_removed_tokens",
                0,
            )
            for item in cleaned_source
        )

        print(
            "    Source overlap removed: "
            f"{removed_tokens} tokens"
        )

        atomic_pieces = (
            atomicize_source_chunks(
                cleaned_source,
                encode,
                decode,
                count,
            )
        )

        content_chunks = pack_content_chunks(
            atomic_pieces,
            count,
        )

        content_chunks = merge_small_chunks(
            content_chunks,
            count,
        )

        final_chunks = build_final_chunks(
            content_chunks,
            encode,
            decode,
            count,
        )

        print(
            f"    Final chunks: "
            f"{len(final_chunks)}"
        )

        all_final.extend(final_chunks)

    final_standards = set(
        get_standards_in_order(all_final)
    )

    validate_chunks(
        all_final,
        count,
        allowed_standards,
    )

    if not final_standards:
        raise RuntimeError(
            "The final output contains no standards."
        )

    if not final_standards.issubset(
        allowed_standards
    ):
        unexpected = (
            final_standards - allowed_standards
        )

        raise RuntimeError(
            "Final output contains unexpected "
            f"standards: {sorted(unexpected)}"
        )

    token_counts = [
        chunk["token_count"]
        for chunk in all_final
    ]

    output = dict(corpus)
    output["metadata"] = dict(
        corpus.get("metadata", {})
    )

    output["metadata"].update(
        {
            "corpus_mode": args.corpus,
            "num_standards": len(
                final_standards
            ),
            "standard_numbers": sorted(
                final_standards
            ),
            "configured_standard_numbers": sorted(
                allowed_standards
            ),
            "num_chunks": len(all_final),
            "skipped_non_standard_chunks": len(
                skipped_chunks
            ),
            "chunking_strategy": (
                "dynamic corpus selection, "
                "non-standard chunk filtering, "
                "OCR-aware standalone-heading splitting, "
                "source-overlap removal, semantic packing "
                "per standard, "
                f"content_target={CONTENT_TARGET}, "
                f"overlap={OVERLAP_TOKENS}, "
                f"max={FINAL_MAX_TOKENS}"
            ),
            "tokenizer": "cl100k_base",
            "content_target_tokens": CONTENT_TARGET,
            "overlap_tokens": OVERLAP_TOKENS,
            "max_tokens": FINAL_MAX_TOKENS,
            "source_overlap_removed": True,
            "embedded_boundaries_split": True,
            "cross_reference_headings_ignored": True,
            "standards_are_hard_boundaries": True,
        }
    )

    output["chunks"] = all_final

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)
    print(
        f"[✓] Standards    : "
        f"{len(final_standards)}"
    )
    print(
        f"[✓] Final chunks : "
        f"{len(all_final)}"
    )
    print(
        f"[✓] Min tokens   : "
        f"{min(token_counts)}"
    )
    print(
        f"[✓] Max tokens   : "
        f"{max(token_counts)}"
    )
    print(
        f"[✓] Avg tokens   : "
        f"{mean(token_counts):.2f}"
    )
    print(
        f"[✓] Median tokens: "
        f"{median(token_counts):.2f}"
    )
    print(
        f"[✓] Saved to     : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()