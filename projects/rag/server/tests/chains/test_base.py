import pytest
from chains.base import get_graph
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from llms import ChatOpenAI, Solar
from retrievers import BM25VectorStore, MetadataVectorStore, TextChunkVectorStore


@pytest.mark.parametrize("llm", [ChatOpenAI(), Solar()])
@pytest.mark.parametrize(
    "retriever",
    [
        TextChunkVectorStore().as_retriever(),
        MetadataVectorStore().as_retriever(),
        BM25VectorStore().as_retriever(),
    ],
)
def test_get_graph(llm: BaseLanguageModel, retriever: BaseRetriever):
    graph = get_graph(llm, retriever)
    output = graph.invoke(
        {
            "user_question": "넌 저기로가",
            "user_character": "론",
        }
    )
    assert output["generation"] is not None
