import logging

import pytest
from retrievers import TextChunkVectorStore, VectorStore


@pytest.fixture(scope="module")
def vectorstore() -> VectorStore:
    return TextChunkVectorStore(chunk_size=1000, chunk_overlap=50)


def test_preprocess(
    caplog: pytest.LogCaptureFixture,
    vectorstore: VectorStore,
) -> None:
    with caplog.at_level(logging.INFO):
        vectorstore.preprocess()
    assert "Saved vectorstore" in caplog.text


def test_as_retriever(
    vectorstore: VectorStore,
) -> None:
    retriever = vectorstore.as_retriever()
    assert retriever is not None
