import argparse
import re

from .pdf_loader import load_pdf


def clean_text(text: str) -> str:
    """Replace repeated whitespace with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split pages into overlapping chunks with source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page in pages:
        text = clean_text(page["text"])
        start = 0
        chunk_number = 1

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_number": chunk_number,
                    "start_char": start,
                    "end_char": end,
                    "text": text[start:end],
                }
            )

            if end == len(text):
                break

            start = end - overlap
            chunk_number += 1

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Split PDF pages into overlapping text chunks."
    )
    parser.add_argument("file_path", help="Path to the PDF file")
    args = parser.parse_args()

    pages = load_pdf(args.file_path)
    chunks = chunk_pages(pages)

    print(f"\nPages containing text: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print(
            f"\n--- {chunk['source']} | Page {chunk['page']} "
            f"| Chunk {chunk['chunk_number']} ---"
        )
        print(f"Characters: {len(chunk['text'])}")
        print(
            f"Position: [{chunk['start_char']}, "
            f"{chunk['end_char']})"
        )
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()