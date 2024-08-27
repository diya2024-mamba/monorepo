import logging
from functools import partial

from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_question: str
    user_character: str
    documents: list[str]
    generation: str


def retrieve(state: GraphState, retriever: BaseRetriever) -> GraphState:
    logging.debug("---RETRIEVE---")
    user_question = state["user_question"]
    user_character = state["user_character"]

    documents = retriever.invoke(user_question, character=user_character)
    logging.debug("Retrieved documents: %s", documents)

    state["documents"] = documents
    return state


def generate(state: GraphState, llm: BaseLanguageModel) -> GraphState:
    logger.debug("---GENERATE---")
    user_character = state["user_character"]
    user_question = state["user_question"]
    documents = state["documents"]

    system_prompt = """
        당신은 흉내를 잘 내는 배우야. 대사를 기반으로 실제 캐릭터처럼 말하는 데 굉장히 능숙해
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "<<<캐릭터>>> \n\n {character} \n\n <<<스크립트>>> \n\n {document} \n\n <<<특징>>> \n\n \
            1. 스크립트를 기반으로 응답하라. \n 2. 대화는 1턴씩 수행하라. \n 3. 캐릭터처럼 이야기하라 \n\n\
            <<<대화>>> {question}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke(
        {
            "character": user_character,
            "document": documents,
            "question": user_question,
        }
    )
    logging.debug("Generated response: %s", generation)

    state["generation"] = generation
    return state


def get_graph(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", partial(retrieve, retriever=retriever))
    workflow.add_node("generate", partial(generate, llm=llm))

    # Define the edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
