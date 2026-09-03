import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModel

from database import get_connection


# CONFIGURATION

MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 5

# Retrieve more candidates than we need, then re-rank down to TOP_K.
# This gives the re-ranker something to actually re-order.
CANDIDATE_K = 15

# Weight given to keyword overlap vs. semantic similarity in final ranking.
# 1.0 = pure semantic similarity, 0.0 = pure keyword overlap.
SEMANTIC_WEIGHT = 0.85

MVP_STANDARD_NUMBERS = (
    8, 9, 10, 11, 12, 13, 17, 19,
    25, 26, 30, 31, 39, 40, 46,
)


# LOAD MODEL

def load_model():

    print("[*] Loading embedding model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"[✗] Failed to load model '{MODEL_NAME}': {e}")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    print(f"[✓] Model loaded on {device}")

    return tokenizer, model, device


# MEAN POOLING

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded, dim=1
    ) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)



# CREATE QUERY EMBEDDING

def create_query_embedding(question, tokenizer, model, device):

    # E5 models are asymmetric: queries and passages need different
    # prefixes. Make sure upload_embeddings.py uses "passage: " when
    # embedding chunks, or similarity scores will be unreliable.
    query_text = "query: " + question

    encoded = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    embedding = mean_pooling(output, encoded["attention_mask"])

    # L2 normalization (required for cosine similarity via pgvector's <=>)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy()[0].tolist()



# VECTOR SEARCH (retrieval)

def fetch_candidates(
    question,
    tokenizer,
    model,
    device,
    corpus="54",
    candidate_k=CANDIDATE_K,
):
    """Retrieve more candidates than needed, so the re-ranker has room to reorder."""

    query_embedding = create_query_embedding(question, tokenizer, model, device)

    vector_string = "[" + ",".join(str(v) for v in query_embedding) + "]"
    standard_filter = ""
    parameters = [vector_string, candidate_k]

    if corpus == "15":
        standard_filter = "WHERE standard_number = ANY(%s)"
        parameters.append(list(MVP_STANDARD_NUMBERS))

    sql = f"""
        SELECT
            chunk_id,
            standard_number,
            standard_title,
            section,
            page_start,
            page_end,
            content,
            metadata,
            similarity
        FROM match_rag_chunks(
            %s::vector,
            %s
        )
        {standard_filter};
    """

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                results = cursor.fetchall()
    except Exception as e:
        print(f"[✗] Database query failed: {e}")
        raise SystemExit(1)

    return results



# RE-RANKING

def tokenize_words(text):
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def rerank(question, candidates, top_k=TOP_K, semantic_weight=SEMANTIC_WEIGHT):
    """
    Combine semantic similarity (from pgvector) with lexical keyword overlap
    to produce a final ranking, then return the top_k results.

    This is a lightweight re-ranker: no extra model calls, cheap to run,
    and helps surface chunks that share exact terminology with the query
    (useful for domain-specific terms like "Murabahah", "Takaful", etc.
    that a purely semantic match can sometimes under-weight).
    """

    question_words = tokenize_words(question)

    scored = []

    for row in candidates:
        content = row[6]  # content column
        similarity = row[8]  # similarity column

        content_words = tokenize_words(content)

        if question_words:
            overlap = len(question_words & content_words) / len(question_words)
        else:
            overlap = 0.0

        final_score = (semantic_weight * similarity) + (
            (1 - semantic_weight) * overlap
        )

        scored.append((final_score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [row for _, row in scored[:top_k]]



# DISPLAY RESULTS

def display_results(question, results):

    print()
    print("=" * 80)
    print(f"VECTOR SEARCH — TOP {len(results)} (re-ranked)")
    print("=" * 80)

    print()
    print(f"Query: {question}")

    print()

    for rank, result in enumerate(results, start=1):

        (
            chunk_id,
            standard_number,
            standard_title,
            section,
            page_start,
            page_end,
            content,
            metadata,
            similarity,
        ) = result

        print("-" * 80)

        print(f"RANK #{rank}")
        print(f"Similarity : {similarity:.4f}")
        print(f"Chunk ID   : {chunk_id}")
        print(f"Standard   : {standard_number}")
        print(f"Title      : {standard_title}")
        print(f"Section    : {section}")
        print(f"Pages      : {page_start} - {page_end}")

        print()
        print("Content:")
        print(content)

    print("-" * 80)


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Search the selected NourFi RAG corpus."
    )
    parser.add_argument(
        "--corpus",
        choices=("15", "54"),
        default="54",
        help="15 for MVP or 54 for the complete corpus.",
    )
    parser.add_argument("--question")
    args = parser.parse_args()

    question = (args.question or input("\nEnter your question: ")).strip()

    if not question:
        print("[!] Question cannot be empty.")
        return

    tokenizer, model, device = load_model()

    print()
    print("[*] Generating query embedding and retrieving candidates...")

    candidates = fetch_candidates(
        question,
        tokenizer,
        model,
        device,
        corpus=args.corpus,
    )

    if not candidates:
        print("[!] No results found.")
        return

    print(f"[✓] Retrieved {len(candidates)} candidates, re-ranking to top {TOP_K}...")

    results = rerank(question, candidates, top_k=TOP_K)

    display_results(question, results)


if __name__ == "__main__":
    main()