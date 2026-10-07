import argparse
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def load_pdf(file_path: str) -> list[dict]:
    """Extract non-empty pages, preserving their source and page number."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    pages = []

    with path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)

        if reader.is_encrypted:
            raise ValueError(
                "This first version does not support encrypted PDFs."
            )

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                print(f"Warning: no text extracted from page {page_number}.")
                continue

            pages.append(
                {
                    "source": path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    if not pages:
        raise ValueError(
            "No readable text found. This PDF may need OCR."
        )

    return pages


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDF text with page references."
    )
    parser.add_argument("file_path", help="Path to the PDF file")
    args = parser.parse_args()

    try:
        pages = load_pdf(args.file_path)
    except (OSError, ValueError, PdfReadError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")

    print(f"\nExtracted text from {len(pages)} page(s).")

    for page in pages:
        print(f"\n--- {page['source']} | Page {page['page']} ---")
        print(page["text"][:500])


if __name__ == "__main__":
    main()