import logging
from functools import partial

from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_question: str
    user_character: str
    documents: list[str]

    search_conversation: str
    character_conversation: str

    ai_character: str
    # movie: str        # 해리포터로 고정
    generation: str


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def grade_documents(state: GraphState, llm: BaseLanguageModel) -> GraphState:
    logging.debug("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    user_character = state['user_character']
    user_question = state['user_question']
    documents = state['documents']

    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "<<<Character>>> {character} \n\n \
                    <<<Retrieved document>>> \n\n {document} \n\n <<<User question>>> {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    filtered_docs = []
    conversation = "Yes"
    for d in documents:
        score = retrieval_grader.invoke(
            {"character": user_character, "question": user_question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            logger.debug("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            conversation = "No"
        else:
            logger.debug("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    state['documents'] = filtered_docs
    state['search_conversation'] = conversation

    return state


def retrieve(state: GraphState, retriever: BaseRetriever) -> GraphState:
    logging.debug("---RETRIEVE---")
    user_question = state["user_question"]

    documents = retriever.invoke(user_question)
    logging.debug("Retrieved documents: %s", documents)

    state["documents"] = documents
    return state


def search_conversation(state: GraphState, llm: BaseLanguageModel) -> GraphState:
    logger.debug("---SEARCH CONVERSATIONS---")
    # movie = state['movie']        # 해리포터로 고정
    movie = "해리포터"
    user_character = state['user_character']

    # Prompt
    conver_system_prompt = """당신은 오덕후이다. 유명한 영화의 캐릭터멸 대사들을 능숙히 발화할 수 있다."""
    conver_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", conver_system_prompt),
            (
                "human",
                "<<<영화>>> {movie} \n\n <<<캐릭터>>> \n\n {character} \n\n \
                <<<대사>>> 캐릭터가 할 만한 대사들을 생성하라.",
            ),
        ]
    )

    conver_generation = conver_prompt | llm | StrOutputParser()
    character_conver = conver_generation.invoke({"movie": movie, "character": user_character})
    state['character_conversation'] = character_conver
    return state


def transform_query(state: GraphState, llm: BaseLanguageModel) -> GraphState:
    logger.debug("---TRANSFORM QUERY---")
    user_question = state["user_question"]
    character_conversation = state['character_conversation']

    # Prompt
    re_write_system_prompt = """당신은 질문을 변형하는 데 능숙하다. 주어친 대사를 참고하여 기존 대사를 변형하여라"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", re_write_system_prompt),
            (
                "human",
                "<<<대사>>> \n\n {character_conver} \n\n \
                <<<기존 대사>>> \n\n {question} \n 개선된 질문의 형태로 변형하라",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    # Re-write question
    better_question = question_rewriter.invoke({"character_conver": character_conversation, "question": user_question})

    state['user_question'] = better_question
    return state


def decide_to_generate(state):
    logger.debug("---ASSESS GRADED DOCUMENTS---")
    state["user_question"]
    conversation = state["search_conversation"]
    state["documents"]

    if conversation == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.debug(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "find_conversation"
    else:
        # We have relevant documents, so generate answer
        logger.debug("---DECISION: GENERATE---")
        return "generate"


def generate(state: GraphState, llm: BaseLanguageModel) -> GraphState:
    logger.debug("---GENERATE---")
    ai_character = state["ai_character"]
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
            "character": ai_character,
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
    workflow.add_node("retrieve", partial(retrieve, retriever=retriever))  # retrieve
    workflow.add_node("grade_documents", partial(grade_documents, llm=llm))  # grade documents
    workflow.add_node("generate", partial(grade_documents, llm=llm))  # generatae
    workflow.add_node("find_conversation", partial(search_conversation, llm=llm))  # search_conversation
    workflow.add_node("transform_query", partial(transform_query, llm=llm))  # transform_query

    # Define the edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "find_conversation": "find_conversation",
            "generate": "generate",
        },
    )
    workflow.add_edge("find_conversation", "transform_query")
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()
