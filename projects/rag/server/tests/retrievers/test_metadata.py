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


def test_search_metadata_faiss(
    vectorstore: MetadataVectorStore,
) -> None:
    result = vectorstore.search_metadata_faiss("script_id", "1")
    value = list(result.values())[0]
    assert value.metadata.get("character") == "해리"
    assert value.page_content == "루모스 막시마! 루모스 막시마!"
