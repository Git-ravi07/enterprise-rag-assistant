import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self, index_dir="data/index"):
        index_path = Path(index_dir)

        self.embeddings = np.load(
            index_path / "embeddings.npy",
            allow_pickle=False,
        )

        with (index_path / "metadata.json").open(
            "r", encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        self.chunks = metadata["chunks"]

        expected_shape = (
            len(self.chunks),
            metadata["dimension"],
        )

        if not self.chunks:
            raise ValueError("The index contains no chunks.")

        if self.embeddings.shape != expected_shape:
            raise ValueError("Embeddings and metadata do not match.")

        if metadata.get("normalized") is not True:
            raise ValueError("This retriever requires normalized embeddings.")

        print("Loading embedding model...")
        self.model = SentenceTransformer(
            metadata["model_name"],
            device="cpu",
        )

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        question = question.strip()

        if not question:
            raise ValueError("Please enter a non-empty question.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        tokens = self.model.tokenizer(
            question,
            add_special_tokens=True,
            truncation=False,
        )

        if len(tokens["input_ids"]) > self.model.max_seq_length:
            raise ValueError("Question is too long. Please shorten it.")

        query_embedding = self.model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ query_embedding
        best_indices = np.argsort(-scores)[:top_k]

        results = []

        for index in best_indices:
            results.append(
                {
                    **self.chunks[int(index)],
                    "score": float(scores[index]),
                }
            )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Search PDF passages by meaning."
    )
    parser.add_argument("question", help="Question in quotation marks")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retriever = SemanticRetriever()
    results = retriever.search(args.question, top_k=args.top_k)

    print(f"\nQuestion: {args.question}")

    for rank, result in enumerate(results, start=1):
        print(
            f"\n--- Result {rank} | {result['source']} "
            f"| Page {result['page']} "
            f"| Chunk {result['chunk_number']} ---"
        )
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])


if __name__ == "__main__":
    main()