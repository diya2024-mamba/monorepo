import random
import time

from chains.base import BaseRAG
from chains.datamodels import (
    CharacterName,
    GradeAnswer,
    GradeDocuments,
    GraphState,
    QueryIntent,
)
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from retrievers.prompt_search import RelationshipSearch

MAX_RETRIEVALS = 3
MAX_GENERATIONS = 3
MIN_SLEEP = 0.5
MAX_SLEEP = 1.5


def random_sleep():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


class AdvancedRAG(BaseRAG):

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        rag_type: str,
    ):
        super().__init__(retriever, llm)
        self.rag_type = rag_type

    def get_graph(self) -> Runnable:
        match self.rag_type:
            case "crag":
                return self._get_corrective_rag_graph()
            case "srag":
                return self._get_self_rag_graph()
            case "prag":
                return self._get_prompt_rag_graph()
            case "base":
                return super().get_graph()

    def _get_self_rag_graph(self) -> Runnable:
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node(
            "find_conversation", self.search_conversation
        )  # search_conversation
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("end_check", self.verification)  # verification

        # Define the edges
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: state["num_retrievals"] < MAX_RETRIEVALS,
            {
                True: "grade_documents",
                False: "generate",
            },
        )
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "find_conversation": "find_conversation",
                "generate": "generate",
            },
        )
        workflow.add_edge("find_conversation", "transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", "end_check")
        workflow.add_conditional_edges(
            "end_check", self.decide_to_end, {"generate": "generate", "end": END}
        )

        return workflow.compile()

    def _get_corrective_rag_graph(self) -> Runnable:
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node(
            "find_conversation", self.search_conversation
        )  # search_conversation
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Define the edges
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: state["num_retrievals"] < MAX_RETRIEVALS,
            {
                True: "grade_documents",
                False: "generate",
            },
        )
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "find_conversation": "find_conversation",
                "generate": "generate",
            },
        )
        workflow.add_edge("find_conversation", "transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()
    def _get_prompt_rag_graph(self) -> Runnable:
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("discriminate_query_intent", self.discriminate_query_intent)  # discriminate query intent
        workflow.add_node("search_relationship", self.search_relationship) # search relationship
        workflow.add_node("small_talk", self.small_talk) # small talk
        workflow.add_node("retrieve", self.retrieve) # retrieve
        workflow.add_node("generate", self.generate)  # generatae

        # Define the edges
        workflow.add_edge(START, "discriminate_query_intent")
        workflow.add_conditional_edges(
            "discriminate_query_intent",
            self.decide_query_intent,
            {
                "search_relationship": "search_relationship",
                "search_story": "retrieve",
                "small_talk": "small_talk"
            },
        )
        workflow.add_edge("search_relationship", "generate")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("small_talk", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def search_relationship(self, state: GraphState):
        self.logger.debug("---SEARCH RELATIONSHIP DOCUMNET---")

        ai_character = state["ai_character"]
        user_question = state["user_question"]

        character_relationship = RelationshipSearch(ai_character)
        name_list = character_relationship.relationship_list

        # LLM with function call
        structred_llm_name_list = self.llm.with_structured_output(CharacterName)

        #Prompt
        rel_system_prompt = """
당신은 판별자입니다.
name_list = {name_list} 주어진 이름 리스트를 읽어라.
user는 {ai_character} 캐릭터와 대화하고 있다.

###지시 사항
1. user의 문장을 읽고 리스트에 있는 이름과 일치하는 원소를 찾아야 한다.
2. 리스트의 원소와 찾는 이름의 문자형이 정확히 일치하지 않을 수 있다.
3. 정답이 한 개다.
4. name_list에 대상이 없을 수 있다. 그러면'없음'을 반환한다.
5. name_list에 정답이 있다면 반드시 name_list의 이름을 한 개 출력한다.
6. 대화하고 있는 {ai_character}는 정답이 될 수 없다.
### example
입력 문장 예시1: 해리 헤르미온느는 어디있어?
입력 문장 예시2: 이봐요 포터, 그레인저 양은 또 어디에 있나요.
입력 문장 예시3: 오늘 헤르미온느 그레인저는 없나?
정답: '헤르미온느 그레인저'

입력 문장 예시1: 크라우치가 범인이야
입력 문장 예시2: 탈옥범 크라우치jr는 어디있나?
정답: '바티미어스 크라우치 주니어'

입력 문장 예시1: 이따 홍길동이랑 같이 밥먹자 어때?
입력 문장 예시2: 10시에 카리나 온대
정답: []
###
"""

        rel_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rel_system_prompt),
                ("human", user_question)
            ]
        )

        name_generation = rel_prompt | structred_llm_name_list
        char_list = name_generation.invoke(
            {
                "ai_character": ai_character,
                "name_list": name_list
            }
        )

        name = char_list.charcater_name
        relationship_doc = []

        if name != "없음":
            doc = name + ": " + character_relationship.search_relationship_document(name)
            relationship_doc.append(doc)
        else:
            relationship_doc = "I don't know."

        state['documents'] = relationship_doc
        return state


    def decide_query_intent(self, state: GraphState):
        self.logger.debug("---DECIDE TO INTENT---")
        query_intent = state["query_intent"]

        if query_intent == "relationship":
            return "search_relationship"
        elif query_intent == "story":
            return "search_story"
        else:
            return "small_talk"

    def small_talk(self, state: GraphState):
        state['documents'] = ["No Document"]
        return state

    def discriminate_query_intent(self, state: GraphState):
        self.logger.debug("---DISCRIMINATE QUERY INTENT---")
        ai_character = state["ai_character"]
        user_question = state["user_question"]
        # Prompt
        disc_system_prompt="""
### System:
You are the discriminator who listens to the user's questions and understands the intent of the sentence. \n
There are three categories in total. \n
'relationship': Asks about the people around the person being asked, or their level of intimacy. \n
'story': Asks about the past experiences or feelings of the person being asked. \n
'small_talk': Requests other light conversations.\n
Choose one of 'relationship' or 'story' or 'small_talk' that matches the intent of the question.
"""
        disc_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", disc_system_prompt),
                    (
                        "human",
                        "### Character: {ai_character} \n\n ### User: {question}",
                    ),]
        )
        # LLM with function call
        structured_llm = self.llm.with_structured_output(QueryIntent)
        disc_generation = disc_prompt | structured_llm
        intent_disc = disc_generation.invoke(
            {
                "ai_character": ai_character,
                "question": user_question
            }
        )
        state['query_intent'] = intent_disc.intent
        return state

    def grade_documents(self, state: GraphState):
        self.logger.debug("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        random_sleep()

        ai_character = state["ai_character"]
        user_question = state["user_question"]
        documents = state["documents"]

        # LLM with function call
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """
### System:
You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "### Character: {ai_character} \n\n ### Retrieved Document: \n {document} \n\n ### User: {question}",
                ),
            ]
        )
        retrieval_grader = grade_prompt | structured_llm_grader

        scores = retrieval_grader.batch(
            [
                {
                    "ai_character": ai_character,
                    "question": user_question,
                    "document": d,
                }
                for d in documents
            ]
        )
        grades = [score.binary_score for score in scores]
        filtered_docs = [d for d, grade in zip(documents, grades) if grade == "yes"]
        if len(filtered_docs) == 0:
            self.logger.debug("---GRADE: DOCUMENT NOT RELEVANT---")
            conversation = "Yes"
        else:
            self.logger.debug("---GRADE: DOCUMENT RELEVANT---")
            conversation = "No"

        state["documents"] = filtered_docs
        state["search_conversation"] = conversation
        return state

    def search_conversation(self, state: GraphState):
        self.logger.debug("---SEARCH CONVERSATIONS---")
        random_sleep()

        # movie = state['movie']        # 해리포터로 고정
        movie = "해리포터"
        user_character = state["user_character"]
        ai_character = state["ai_character"]

        # Prompt
        conver_system_prompt = """### System: \n You are a movie buff. You can skillfully recite lines from famous movie characters."""
        conver_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", conver_system_prompt),
                (
                    "human",
                    "### Movie Title: \n {movie} \n\n ## Character Name: \n {ai_character} \n\n \
                    ### Instruction: \n Generate lines that {ai_character} would say to {user_character}. Speak 한국어.",
                ),
            ]
        )

        conver_generation = conver_prompt | self.llm | StrOutputParser()
        character_conver = conver_generation.invoke(
            {
                "movie": movie,
                "user_character": user_character,
                "ai_character": ai_character,
            }
        )
        state["character_conversation"] = character_conver
        return state

    def transform_query(self, state: GraphState):
        self.logger.debug("---TRANSFORM QUERY---")
        random_sleep()

        user_question = state["user_question"]
        character_conversation = state["character_conversation"]

        # Prompt
        re_write_system_prompt = """### System: \n You are skilled at modifying questions. Based on the given lines, transform the original dialogue accordingly."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", re_write_system_prompt),
                (
                    "human",
                    "### Scripts \n {character_conver} \n\n ### Original Sentence \n {question} \n\n ### Instruction \n Transform the original dialogue into an improved question form. Speak 한국어.",
                ),
            ]
        )

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        # Re-write question
        better_question = question_rewriter.invoke(
            {"character_conver": character_conversation, "question": user_question}
        )

        state["user_question"] = better_question
        return state

    def decide_to_generate(self, state: GraphState):
        self.logger.debug("---ASSESS GRADED DOCUMENTS---")
        conversation = state["search_conversation"]

        if conversation == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            self.logger.debug(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "find_conversation"
        else:
            # We have relevant documents, so generate answer
            self.logger.debug("---DECISION: GENERATE---")
            return "generate"

    def verification(self, state: GraphState):
        self.logger.debug("---Answer Check---")
        random_sleep()

        user_question = state["user_question"]
        generation = state["generation"]

        # LLM with function call
        structured_llm = self.llm.with_structured_output(GradeAnswer)

        review_system_prompt = """### System: \n You are a skilled communicator who excels at understanding the context of conversations. Review the generated dialogue to ensure it aligns with the flow of the conversation or introduces creative elements."""
        review_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", review_system_prompt),
                (
                    "human",
                    """
### User: {question}
### Assistant: {generation}

### Instruction:
- Review the above conversation to check if the Assitant's response aligns with the flow or introduces creativity.
- If the Assistant's response is appropriate or creative, generate "yes"; if not, generate "no."
""",
                ),
            ]
        )

        answer_checker = review_prompt | structured_llm
        answer = answer_checker.invoke(
            {"question": user_question, "generation": generation}
        )
        if answer == "yes":
            self.logger.debug("---GOOD GENERATION---")
            check = "yes"
        else:
            self.logger.debug("---BAD GENERATION---")
            check = "no"

        state["answer_check"] = check
        return state

    def decide_to_end(self, state: GraphState):
        self.logger.debug("---ASSESS GENERATION---")
        if (
            state["num_retrievals"] >= MAX_RETRIEVALS
            or state["num_generations"] >= MAX_GENERATIONS
        ):
            return "end"

        check = state["answer_check"]

        if check == "Yes":
            return "end"
        else:
            return "generate"
