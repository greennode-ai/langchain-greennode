"""Test chat model integration."""

import pytest
from langchain_core.documents import Document

from langchain_greennode import GreenNodeRerank


def test_initialization() -> None:
    """Test chat model initialization."""
    GreenNodeRerank()


def test_doc_to_string_with_string() -> None:
    """Test document serialization"""
    reranker = GreenNodeRerank()
    out = reranker._document_to_str("test str")
    assert out == "test str"


def test_doc_to_string_with_document() -> None:
    """Test document serialization"""
    reranker = GreenNodeRerank()
    out = reranker._document_to_str(Document(page_content="test str"))
    assert out == "test str"


def test_doc_to_string_with_dict() -> None:
    """Test document serialization"""
    reranker = GreenNodeRerank()
    out = reranker._document_to_str({"title": "hello", "text": "test str"})
    assert out == "title: hello\ntext: test str\n"


def test_doc_to_string_with_dicts_with_rank_fields() -> None:
    """Test document serialization"""
    reranker = GreenNodeRerank()
    out = reranker._document_to_str(
        document={"title": "hello", "text": "test str"}, rank_fields=["text"]
    )
    assert out == "text: test str\n"


def test_greennode_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        GreenNodeRerank(model_kwargs={"model": "foo"})


def test_greennode_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = GreenNodeRerank(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}

