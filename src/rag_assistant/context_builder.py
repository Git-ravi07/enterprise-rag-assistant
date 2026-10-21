
def build_context(results: list[dict]) -> str:
    """
    Convert retrieved document chunks into
    numbered evidence blocks for an LLM.
    """

    if not results:
        return ""

    evidence_blocks = []

    for index, result in enumerate(results, start=1):
        source = result["source"]
        page = result["page"]
        text = result["text"].strip()

        if not text:
            continue

        block = (
            f"[S{index}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

        evidence_blocks.append(block)

    return "\n\n".join(evidence_blocks)


if __name__ == "__main__":
    sample_results = [
        {
            "source": "sample.pdf",
            "page": 3,
            "text": "The system uses Qdrant.",
        },
        {
            "source": "sample.pdf",
            "page": 8,
            "text": "Qdrant performs vector similarity search.",
        },
    ]

    context = build_context(sample_results)
    print(context)
