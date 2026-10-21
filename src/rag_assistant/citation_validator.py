
import re


def validate_citations(answer: str, results: list[dict]) -> dict:
    """
    Validate citation references against retrieved passages.

    Supports:
      [S1]
      [S1, S2]
      [S1, S2, S3]
    """

    # Find all bracketed citation groups.
    groups = re.findall(
        r"\[(S\d+(?:\s*,\s*S\d+)*)\]",
        answer
    )

    cited_ids = []

    for group in groups:
        ids = re.findall(r"S(\d+)", group)

        for source_id in ids:
            cited_ids.append(int(source_id))

    # Remove duplicates while preserving order.
    cited_ids = list(dict.fromkeys(cited_ids))

    valid_ids = [
        i for i in cited_ids
        if 1 <= i <= len(results)
    ]

    invalid_ids = [
        i for i in cited_ids
        if i < 1 or i > len(results)
    ]

    return {
        "valid": len(invalid_ids) == 0 and len(valid_ids) > 0,
        "cited_ids": valid_ids,
        "invalid_ids": invalid_ids,
        "citations_found": len(cited_ids)
    }
