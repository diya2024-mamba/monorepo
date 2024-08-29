from chains.base import BaseRAG
from chains.datamodels import GradeAnswer, GradeDocuments, GraphState
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

MAX_RETRIEVALS = 3


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

    def grade_documents(self, state: GraphState):
        self.logger.debug("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

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
        if state["num_retrievals"] >= MAX_RETRIEVALS:
            return "end"

        check = state["answer_check"]

        if check == "Yes":
            return "end"
        else:
            return "generate"
