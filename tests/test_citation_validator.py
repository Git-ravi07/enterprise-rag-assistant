
from src.rag_assistant.citation_validator import validate_citations


def test_single_citation():
    results = [{"text": "Qdrant"}]

    answer = "The database is Qdrant [S1]."

    report = validate_citations(answer, results)

    assert report["valid"] is True
    assert report["cited_ids"] == [1]


def test_grouped_citations():
    results = [
        {"text": "First passage"},
        {"text": "Second passage"},
        {"text": "Third passage"},
    ]

    answer = "The database is Qdrant [S1, S2, S3]."

    report = validate_citations(answer, results)

    assert report["valid"] is True
    assert report["cited_ids"] == [1, 2, 3]


def test_invalid_citation():
    results = [{"text": "Only one passage"}]

    answer = "The answer is Qdrant [S5]."

    report = validate_citations(answer, results)

    assert report["valid"] is False
    assert report["invalid_ids"] == [5]


def test_missing_citation():
    results = [{"text": "Qdrant"}]

    answer = "The database is Qdrant."

    report = validate_citations(answer, results)

    assert report["valid"] is False
