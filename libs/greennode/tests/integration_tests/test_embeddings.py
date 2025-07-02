"""Test GreenNode Serverless AI embeddings."""

import pytest
from langchain_greennode import GreenNodeEmbeddings


def test_langchain_greennode_embed_documents() -> None:
    """Test GreenNode Serverless AI embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = GreenNodeEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_greennode_embed_query() -> None:
    """Test GreenNode Serverless AI embeddings."""
    query = "foo bar"
    embedding = GreenNodeEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0

@pytest.mark.asyncio
async def test_langchain_greennode_aembed_documents() -> None:
    """Test GreenNode Serverless AI embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = GreenNodeEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0

@pytest.mark.asyncio
async def test_langchain_greennode_aembed_query() -> None:
    """Test GreenNode Serverless AI embeddings asynchronous."""
    query = "foo bar"
    embedding = GreenNodeEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0