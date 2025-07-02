
from langchain_core.documents import Document

from langchain_greennode import GreenNodeRerank


def test_langchain_greennode_rerank_documents() -> None:
    rerank = GreenNodeRerank()
    test_documents = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document."),
    ]
    test_query = "Test query"
    results = rerank.rerank(test_documents, test_query)
    assert len(results) == 2


def test_langchain_greennode_rerank_with_rank_fields() -> None:
    rerank = GreenNodeRerank()
    test_documents = [
        {"content": "This document is about Penguins.", "subject": "Physics"},
        {"content": "This document is about Physics.", "subject": "Penguins"},
    ]
    test_query = "penguins"

    response = rerank.rerank(test_documents, test_query, rank_fields=["content"])

    assert len(response) == 2
    assert response[0]["index"] == 0
    results = {r["index"]: r["relevance_score"] for r in response}
    assert results[0] > results[1]