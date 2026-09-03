"""Run all NourFi RAG steps in order.

Run from the project root:
    python App\\AI-RAG\\rag\\run_pipeline.py --corpus 54
    python App\\AI-RAG\\rag\\run_pipeline.py --corpus 15 --question "What is Murabahah?"
"""

import argparse
import subprocess
import sys
from pathlib import Path


RAG_DIR = Path(__file__).resolve().parent

BUILD_STEPS = (
    ("Chunking", "chunker.py"),
    ("Embeddings", "embeddings.py"),
    ("Upload", "upload_embeddings.py"),
)

QUERY_STEPS = (
    ("Vector search", "vector_search.py"),
    ("Grounded answer", "pipeline.py"),
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the complete NourFi RAG pipeline."
    )
    parser.add_argument(
        "--corpus",
        choices=("15", "54"),
        help="Corpus used by every step: 15 or 54.",
    )
    parser.add_argument(
        "--question",
        help="Question sent to vector search and pipeline.",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Run chunking, embeddings, and upload only.",
    )
    return parser.parse_args()


def prompt_for_corpus():
    while True:
        corpus = input("Select corpus [15/54]: ").strip()
        if corpus in ("15", "54"):
            return corpus
        print("Please enter 15 or 54.")


def prompt_for_question():
    while True:
        question = input("Enter your question: ").strip()
        if question:
            return question
        print("Question cannot be empty.")


def run_step(label, script_name, arguments):
    script_path = RAG_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing pipeline step: {script_path}")

    command = [sys.executable, str(script_path), *arguments]

    print()
    print("=" * 70)
    print(f"[STEP] {label}: {script_name}")
    print("=" * 70)

    try:
        subprocess.run(command, cwd=str(RAG_DIR), check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} failed with exit code {error.returncode}."
        ) from error

    print(f"[OK] {label} complete.")


def main():
    args = parse_arguments()
    corpus = args.corpus or prompt_for_corpus()

    print()
    print("=" * 70)
    print(f"NOURFI RAG PIPELINE - CORPUS {corpus}")
    print("=" * 70)

    for label, script_name in BUILD_STEPS:
        run_step(label, script_name, ["--corpus", corpus])

    if args.skip_query:
        print("\nBuild complete. Query steps skipped.")
        return

    question = args.question or prompt_for_question()

    for label, script_name in QUERY_STEPS:
        run_step(
            label,
            script_name,
            ["--corpus", corpus, "--question", question],
        )

    print()
    print("=" * 70)
    print("ALL PIPELINE STEPS COMPLETE")
    print(f"Corpus: {corpus}")
    print(f"Question: {question}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
    except Exception as error:
        sys.exit(f"\nPipeline stopped: {error}")
