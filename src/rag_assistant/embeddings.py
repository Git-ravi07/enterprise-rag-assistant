import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import chunk_pages
from .pdf_loader import load_pdf


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def prepare_chunks(pages, model):
    """Find a character chunk size that fits the model's token limit."""
    for size in (1000, 750, 500, 250, 100):
        overlap = size // 5
        chunks = chunk_pages(
            pages,
            chunk_size=size,
            overlap=overlap,
        )

        texts = [chunk["text"] for chunk in chunks]

        tokens = model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

        largest = max(len(ids) for ids in tokens["input_ids"])

        if largest <= model.max_seq_length:
            print(f"Chunk size: {size} characters")
            print(f"Overlap: {overlap} characters")
            print(f"Largest chunk: {largest} tokens")
            return chunks

    raise ValueError("Unable to fit chunks within the model's token limit.")


def main():
    parser = argparse.ArgumentParser(
        description="Create and save embeddings for a PDF."
    )
    parser.add_argument("file_path", help="Path to the PDF file")
    args = parser.parse_args()

    pages = load_pdf(args.file_path)

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    chunks = prepare_chunks(pages, model)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    output_dir = Path("data/index")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)

    metadata = {
        "model_name": MODEL_NAME,
        "normalized": True,
        "dimension": int(embeddings.shape[1]),
        "chunks": chunks,
    }

    with (output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First vector length: {np.linalg.norm(embeddings[0]):.4f}")
    print(f"Saved index to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()