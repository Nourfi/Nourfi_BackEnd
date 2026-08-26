"""
NourFi — RAG Embeddings

Free local embeddings using:
    intfloat/multilingual-e5-small

Usage:

    python App\\AI-RAG\\rag\\embeddings.py --corpus 15
    python App\\AI-RAG\\rag\\embeddings.py  --corpus 54

"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# PATHS

BASE_DIR = Path(__file__).resolve().parents[1]

CORPUS_CONFIG = {
    "15": {
        "input": (
            BASE_DIR
            / "data"
            / "processed"
            / "aaoifi_rag_chunks_15_standards_500.json"
        ),
        "embeddings_output": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embeddings_15_standards.npy"
        ),
        "metadata_output": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embedding_metadata_15_standards.json"
        ),
    },

    "54": {
        "input": (
            BASE_DIR
            / "data"
            / "processed"
            / "aaoifi_rag_chunks_54_standards_500.json"
        ),
        "embeddings_output": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embeddings_54_standards.npy"
        ),
        "metadata_output": (
            BASE_DIR
            / "data"
            / "embeddings"
            / "aaoifi_embedding_metadata_54_standards.json"
        ),
    },
}


# MODEL

MODEL_NAME = "intfloat/multilingual-e5-small"

BATCH_SIZE = 8

MAX_LENGTH = 512


# ARGUMENTS

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Generate NourFi RAG embeddings."
    )

    parser.add_argument(
        "--corpus",
        choices=("15", "54"),
        default="15",
        help="15 for MVP or 54 for complete corpus.",
    )

    return parser.parse_args()


# LOAD MODEL

def load_embedding_model():

    print()
    print("[*] Loading embedding model...")
    print(f"    {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    # CPU is used automatically.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(
        f"[✓] Model loaded on: {device}"
    )

    return tokenizer, model, device

# MEAN POOLING

def mean_pooling(
    model_output,
    attention_mask,
):

    token_embeddings = (
        model_output.last_hidden_state
    )

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings
        * input_mask_expanded,
        dim=1,
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9,
    )


# EMBEDDING

def generate_embeddings(
    tokenizer,
    model,
    device,
    chunks,
):

    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    all_embeddings = []

    total = len(texts)

    print()
    print("[*] Generating embeddings...")
    print(f"[*] Total chunks: {total}")
    print(f"[*] Batch size: {BATCH_SIZE}")
    print()

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = texts[start:end]

        print(
            f"    Embedding chunks "
            f"{start + 1}-{end}/{total}"
        )

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            model_output = model(
                **encoded
            )

        embeddings = mean_pooling(
            model_output,
            encoded["attention_mask"],
        )

        # Normalize embeddings.
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
        )

        all_embeddings.append(
            embeddings
        )

    return np.vstack(
        all_embeddings
    ).astype(
        np.float32
    )

# SAVE

def save_results(
    embeddings,
    chunks,
    embeddings_path,
    metadata_path,
):

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save embeddings

    np.save(
        embeddings_path,
        embeddings,
    )

    # Save metadata

    metadata = []

    for index, chunk in enumerate(
        chunks
    ):

        metadata.append(
            {
                "vector_index": index,

                "chunk_id": chunk.get(
                    "chunk_id"
                ),

                "standard_number": chunk.get(
                    "standard_number"
                ),

                "standard_title": chunk.get(
                    "standard_title"
                ),

                "section": chunk.get(
                    "section"
                ),

                "page_start": chunk.get(
                    "page_start"
                ),

                "page_end": chunk.get(
                    "page_end"
                ),

                "source_chunk_ids": chunk.get(
                    "source_chunk_ids",
                    [],
                ),

                "source_id": chunk.get(
                    "source_id"
                ),

                "source_type": chunk.get(
                    "source_type"
                ),

                "language": chunk.get(
                    "language"
                ),

                "content_token_count": chunk.get(
                    "content_token_count"
                ),

                "overlap_token_count": chunk.get(
                    "overlap_token_count"
                ),

                "token_count": chunk.get(
                    "token_count"
                ),

                "text": chunk.get(
                    "text"
                ),
            }
        )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

# MAIN

def main():

    args = parse_arguments()

    config = CORPUS_CONFIG[
        args.corpus
    ]

    print("=" * 70)
    print(
        f"NOURFI — RAG EMBEDDINGS "
        f"({args.corpus}-STANDARD MODE)"
    )
    print("=" * 70)

    print(
        f"[*] Model: {MODEL_NAME}"
    )

    print(
        f"[*] Input: {config['input']}"
    )

    # Load chunks


    if not config["input"].exists():

        raise FileNotFoundError(
            f"Chunk file not found:\n"
            f"{config['input']}"
        )

    print()
    print("[*] Loading chunks...")

    with open(
        config["input"],
        "r",
        encoding="utf-8",
    ) as file:

        corpus = json.load(file)

    chunks = corpus.get(
        "chunks",
        [],
    )

    if not chunks:

        raise RuntimeError(
            "No chunks found in corpus."
        )

    print(
        f"[✓] Loaded {len(chunks)} chunks"
    )

    # Load model

    tokenizer, model, device = (
        load_embedding_model()
    )

    # Generate embeddings
    embeddings = generate_embeddings(
        tokenizer,
        model,
        device,
        chunks,
    )

    # Validate

    if len(embeddings) != len(chunks):

        raise RuntimeError(
            "Number of embeddings does not "
            "match number of chunks."
        )

    print()
    print(
        f"[✓] Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    # Save

    save_results(
        embeddings,
        chunks,
        config["embeddings_output"],
        config["metadata_output"],
    )

    # Final report


    print()
    print("=" * 70)
    print("EMBEDDING COMPLETE")
    print("=" * 70)

    print(
        f"[✓] Corpus      : "
        f"{args.corpus} standards"
    )

    print(
        f"[✓] Chunks      : "
        f"{len(chunks)}"
    )

    print(
        f"[✓] Dimensions  : "
        f"{embeddings.shape[1]}"
    )

    print(
        f"[✓] Embeddings  : "
        f"{config['embeddings_output']}"
    )

    print(
        f"[✓] Metadata    : "
        f"{config['metadata_output']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()