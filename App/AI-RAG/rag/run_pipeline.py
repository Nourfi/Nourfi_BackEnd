"""NourFi RAG pipeline runner.

Examples:

    Ask a question using the existing Supabase dataset:
        python App\\AI-RAG\\rag\\run_pipeline.py --corpus 54 --question "What is Standard 24 about?" --persona general_public

    Rebuild and upload after changing source documents:
        python App\\AI-RAG\\rag\\run_pipeline.py --corpus 54 --force-rebuild
        -> This will run the chunker, embeddings, and upload steps.
        The rebuild pipeline should run : 
        during deployment;
        after changing corpus documents;
        manually when rebuilding the database.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR.parent / "data"

BUILD_STEPS = (
    ("Chunking", "chunker.py"),
    ("Embeddings", "embeddings.py"),
    ("Upload", "upload_embeddings.py"),
)


def validate_runtime() -> None:
    checks = (
        ("numpy", "load"),
        ("tiktoken", "get_encoding"),
        ("torch", "no_grad"),
        ("transformers", "AutoTokenizer"),
    )

    for module_name, required_attribute in checks:
        try:
            module = __import__(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"Missing dependency '{module_name}'. Run:\n"
                f"{sys.executable} -m pip install -r requirements.txt"
            ) from error

        if not hasattr(module, required_attribute):
            location = getattr(module, "__file__", "unknown location")
            raise RuntimeError(
                f"Invalid '{module_name}' import from {location}.\n"
                f"Expected attribute: '{required_attribute}'.\n"
                f"Imported from: {location}\n\n"
                f"Repair the environment with:\n"
                f"{sys.executable} -m pip install "
                f"--force-reinstall -r requirements.txt"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NourFi RAG build and query pipeline."
    )

    parser.add_argument(
        "--corpus",
        choices=("15", "54"),
        default="54",
        help="Corpus to use. Default: 54.",
    )

    parser.add_argument(
        "--question",
        help="Question to ask after the build phase.",
    )

    parser.add_argument(
        "--persona",
        default="general_public",
        help="Persona used to format the answer.",
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build and upload the local corpus before querying.",
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Regenerate chunks and embeddings, then upload them. "
            "Use only after changing the source documents."
        ),
    )

    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Stop after an explicitly requested build.",
    )

    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help=(
            "Regenerate local chunks/embeddings but do not upload them. "
            "Useful for local validation."
        ),
    )

    return parser.parse_args()


def artifact_paths(corpus: str) -> tuple[Path, Path, Path]:
    return (
        DATA_DIR
        / "processed"
        / f"aaoifi_rag_chunks_{corpus}_standards_500.json",
        DATA_DIR
        / "embeddings"
        / f"aaoifi_embeddings_{corpus}_standards.npy",
        DATA_DIR
        / "embeddings"
        / f"aaoifi_embedding_metadata_{corpus}_standards.json",
    )


def artifacts_exist(corpus: str) -> bool:
    return all(path.is_file() for path in artifact_paths(corpus))


def run_step(
    label: str,
    script_name: str,
    arguments: list[str],
) -> None:
    script_path = RAG_DIR / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Missing pipeline step: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        *arguments,
    ]

    print()
    print("=" * 70)
    print(f"[STEP] {label}")
    print("=" * 70)

    try:
        subprocess.run(
            command,
            cwd=str(RAG_DIR),
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} failed with exit code {error.returncode}."
        ) from error

    print(f"[OK] {label} complete.")


def build_corpus(
    corpus: str,
    force_rebuild: bool,
    skip_upload: bool,
) -> None:
    if artifacts_exist(corpus) and not force_rebuild:
        print(
            "[OK] Existing chunks and embeddings found."
        )
        print(
            "[OK] Build and upload skipped."
        )
        return

    if force_rebuild:
        print(
            "[INFO] Force rebuild enabled."
        )
    else:
        print(
            "[INFO] Required artifacts are missing."
        )

    run_step(
        "Chunking",
        "chunker.py",
        ["--corpus", corpus],
    )

    run_step(
        "Embeddings",
        "embeddings.py",
        ["--corpus", corpus],
    )

    if skip_upload:
        print(
            "[INFO] Upload skipped because --skip-upload was supplied."
        )
        return

    run_step(
        "Upload",
        "upload_embeddings.py",
        ["--corpus", corpus],
    )


def run_query(
    corpus: str,
    question: str | None,
    persona: str,
) -> None:
    if not question:
        question = input("Enter your question: ").strip()

    if not question:
        raise RuntimeError("Question cannot be empty.")

    run_step(
        "Grounded answer",
        "pipeline.py",
        [
            "--corpus",
            corpus,
            "--question",
            question,
            "--persona",
            persona,
        ],
    )


def main() -> None:
    args = parse_arguments()
    validate_runtime()

    print()
    print("=" * 70)
    print(f"NOURFI RAG PIPELINE - CORPUS {args.corpus}")
    print("=" * 70)

    if args.build or args.force_rebuild or args.skip_upload or args.skip_query:
        build_corpus(
            corpus=args.corpus,
            force_rebuild=args.force_rebuild,
            skip_upload=args.skip_upload,
        )
    else:
        print("[OK] Using the existing Supabase corpus; local build skipped.")

    if args.skip_query:
        print()
        print("Build phase complete.")
        print("No question was executed.")
        return

    run_query(
        corpus=args.corpus,
        question=args.question,
        persona=args.persona,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
    except Exception as error:
        sys.exit(f"\nPipeline stopped: {error}")