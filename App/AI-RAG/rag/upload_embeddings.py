"""
NourFi — Upload Embeddings to Supabase

Usage:

    python App\\AI-RAG\\rag\\upload_embeddings.py --corpus 15
    python App\\AI-RAG\\rag\\upload_embeddings.py --corpus 54
"""

import argparse
import json
from pathlib import Path

import numpy as np

from database import get_connection


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CORPUS_CONFIG = {
    "15": {
        "embeddings": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embeddings_15_standards.npy"
        ),
        "metadata": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embedding_metadata_15_standards.json"
        ),
    },
    "54": {
        "embeddings": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embeddings_54_standards.npy"
        ),
        "metadata": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embedding_metadata_54_standards.json"
        ),
    },
}


TABLE_NAME = "rag_chunks"

BATCH_SIZE = 50


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Upload NourFi RAG embeddings to Supabase."
    )

    parser.add_argument(
        "--corpus",
        choices=("15", "54"),
        default="15",
        help="15 for MVP or 54 for complete corpus.",
    )

    return parser.parse_args()


# ============================================================
# LOAD DATA
# ============================================================

def load_data(embeddings_path, metadata_path):

    print("[*] Loading embeddings...")

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found:\n{embeddings_path}"
        )

    embeddings = np.load(
        embeddings_path
    )

    print(
        f"[✓] Embeddings: {embeddings.shape}"
    )

    print("[*] Loading metadata...")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"[✓] Metadata: {len(metadata)}"
    )

    if len(embeddings) != len(metadata):

        raise RuntimeError(
            "Embeddings count does not match metadata count."
        )

    if embeddings.shape[1] != 384:

        raise RuntimeError(
            f"Expected 384 dimensions, "
            f"got {embeddings.shape[1]}"
        )

    return embeddings, metadata


# ============================================================
# BUILD ROW
# ============================================================

def build_row(embedding, item):

    return (
        item["chunk_id"],
        item.get("standard_number"),
        item.get("standard_title"),
        item.get("section"),
        item.get("page_start"),
        item.get("page_end"),
        json.dumps(
            item.get("source_chunk_ids", []),
            ensure_ascii=False
        ),
        item.get("source_id"),
        item.get("source_type"),
        item.get("language"),
        item.get("content_token_count"),
        item.get("overlap_token_count"),
        item.get("token_count"),
        item["text"],
        embedding.tolist(),
        json.dumps(
            {
                "vector_index": item["vector_index"]
            },
            ensure_ascii=False
        )
    )


# ============================================================
# UPLOAD
# ============================================================

def upload_embeddings(corpus):

    config = CORPUS_CONFIG[corpus]

    embeddings, metadata = load_data(
        config["embeddings"],
        config["metadata"],
    )

    total = len(metadata)

    print()
    print("=" * 70)
    print(f"UPLOADING EMBEDDINGS TO SUPABASE ({corpus}-STANDARD MODE)")
    print("=" * 70)

    print(f"[*] Total rows: {total}")
    print(f"[*] Batch size: {BATCH_SIZE}")

    insert_query = """
        INSERT INTO rag_chunks (
            chunk_id,
            standard_number,
            standard_title,
            section,
            page_start,
            page_end,
            source_chunk_ids,
            source_id,
            source_type,
            language,
            content_token_count,
            overlap_token_count,
            token_count,
            content,
            embedding,
            metadata
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s::jsonb,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s::vector,
            %s::jsonb
        )
        ON CONFLICT (chunk_id)
        DO UPDATE SET
            standard_number = EXCLUDED.standard_number,
            standard_title = EXCLUDED.standard_title,
            section = EXCLUDED.section,
            page_start = EXCLUDED.page_start,
            page_end = EXCLUDED.page_end,
            source_chunk_ids = EXCLUDED.source_chunk_ids,
            source_id = EXCLUDED.source_id,
            source_type = EXCLUDED.source_type,
            language = EXCLUDED.language,
            content_token_count = EXCLUDED.content_token_count,
            overlap_token_count = EXCLUDED.overlap_token_count,
            token_count = EXCLUDED.token_count,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata;
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            for start in range(
                0,
                total,
                BATCH_SIZE
            ):

                end = min(
                    start + BATCH_SIZE,
                    total
                )

                rows = []

                for index in range(
                    start,
                    end
                ):

                    row = build_row(
                        embeddings[index],
                        metadata[index]
                    )

                    rows.append(row)

                print(
                    f"[*] Uploading "
                    f"{start + 1}-{end}/{total}"
                )

                cursor.executemany(
                    insert_query,
                    rows
                )

                connection.commit()

                print(
                    f"[✓] Uploaded {len(rows)} rows"
                )

    print()
    print("=" * 70)
    print("UPLOAD COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_arguments()
    upload_embeddings(args.corpus)


if __name__ == "__main__":
    main()