"""Test embedding model integration."""

import os

import pytest  # type: ignore[import-not-found]

from langchain_greennode import GreenNodeEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    GreenNodeEmbeddings()


def test_greennode_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        GreenNodeEmbeddings(model_kwargs={"model": "foo"})


def test_greennode_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = GreenNodeEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}