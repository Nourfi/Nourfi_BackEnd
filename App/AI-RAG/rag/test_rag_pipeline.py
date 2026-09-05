import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR.parent / "data"


def load_runner():
	spec = importlib.util.spec_from_file_location("run_pipeline", RAG_DIR / "run_pipeline.py")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def test_complete_corpus_artifacts_are_aligned():
	if not all(path.is_file() for path in (
		DATA_DIR / "processed" / "aaoifi_rag_chunks_54_standards_500.json",
		DATA_DIR / "embeddings" / "aaoifi_embeddings_54_standards.npy",
		DATA_DIR / "embeddings" / "aaoifi_embedding_metadata_54_standards.json",
	)):
		pytest.skip("Local corpus artifacts are optional; Supabase is the runtime source.")

	chunks = json.loads(
		(DATA_DIR / "processed" / "aaoifi_rag_chunks_54_standards_500.json")
		.read_text(encoding="utf-8")
	)["chunks"]
	metadata = json.loads(
		(DATA_DIR / "embeddings" / "aaoifi_embedding_metadata_54_standards.json")
		.read_text(encoding="utf-8")
	)
	embeddings = np.load(
		DATA_DIR / "embeddings" / "aaoifi_embeddings_54_standards.npy",
		mmap_mode="r",
	)

	assert len(chunks) == len(metadata) == embeddings.shape[0]
	assert embeddings.shape[1] == 384
	assert {item["standard_number"] for item in metadata} == set(range(1, 55))


def test_runner_exposes_optional_local_artifact_check():
	runner = load_runner()

	assert callable(runner.artifacts_exist)


def test_runner_help_is_available_without_model_or_database():
	result = subprocess.run(
		[sys.executable, str(RAG_DIR / "run_pipeline.py"), "--help"],
		capture_output=True,
		text=True,
		check=True,
	)

	assert "--force-rebuild" in result.stdout
	assert "--skip-query" in result.stdout
