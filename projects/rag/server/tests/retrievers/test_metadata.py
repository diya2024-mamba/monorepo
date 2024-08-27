import logging

import pytest
from retrievers import MetadataVectorStore, VectorStore


@pytest.fixture(scope="module")
def vectorstore() -> VectorStore:
    return MetadataVectorStore()


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


def test_invoke(
    vectorstore: MetadataVectorStore,
) -> None:
    retriever = vectorstore.as_retriever()
    output = retriever.invoke("안녕?", character="해리")
    assert isinstance(output, list)
    assert len(output) > 0
