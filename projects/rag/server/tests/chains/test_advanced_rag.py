import pytest
from chains.advanced_rag import AdvancedRAG
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from llms import GPT4o, Llama3_1
from retrievers import BM25VectorStore, MetadataVectorStore, TextChunkVectorStore


@pytest.mark.parametrize("llm", [GPT4o, Llama3_1])
@pytest.mark.parametrize(
    "retriever",
    [
        TextChunkVectorStore().as_retriever(),
        MetadataVectorStore().as_retriever(),
        BM25VectorStore().as_retriever(),
    ],
)
@pytest.mark.parametrize("method", ["base", "crag", "srag"])
def test_get_graph(llm: BaseLanguageModel, retriever: BaseRetriever, method: str):
    rag = AdvancedRAG(retriever, llm, method)
    graph = rag.get_graph()
    output = graph.invoke(
        {
            "user_question": "넌 저기로가",
            "user_character": "론",
            "ai_character": "말포이",
        },
        config={"recursion_limit": 30},
    )
    assert output["generation"] is not None
