import logging

from chains.datamodels import GraphState
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph


class BaseRAG:
    def __init__(self, retriever: BaseRetriever, llm: BaseLanguageModel):
        self.retriever = retriever
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def retrieve(self, state: GraphState):
        self.logger.debug("---RETRIEVE---")
        user_question = state["user_question"]
        user_character = state["user_character"]

        if state.get("num_retrievals") is None:
            state["num_retrievals"] = 1
        else:
            state["num_retrievals"] += 1
        self.logger.debug("Number of retrievals: %s", state["num_retrievals"])

        documents = self.retriever.invoke(user_question, character=user_character)
        self.logger.debug("Retrieved documents: %s", documents)

        state["documents"] = documents
        return state

    def generate(self, state: GraphState, temperature: float = None):
        self.logger.debug("---GENERATE---")
        user_character = state["user_character"]
        ai_character = state["ai_character"]
        user_question = state["user_question"]
        documents = "\n".join(state["documents"])
        if temperature is not None:
            self.llm.temperature = temperature
        system_prompt = "Please try to provide useful, helpful and actionable answers."
        user_prompt = """Act as {ai_character} in movie 해리포터. Below is the scripts of {ai_character} from the movie you can refer to.
{document}

You are currently having conversation with {user_character}. Response should be maximum 2 sentences. 한국어로 대답하세요.

User: {question}"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        chain_input = {
            "user_character": user_character,
            "ai_character": ai_character,
            "document": documents,
            "question": user_question,
        }
        self.logger.debug(f"Input:{prompt.format_messages(**chain_input)}")
        chain = prompt | self.llm | StrOutputParser()
        generation = chain.invoke(chain_input)
        if generation.startswith("#") and ":" in generation:
            generation = generation.split(":")[1]

        self.logger.debug("Generated response: %s", generation)

        state["generation"] = generation
        return state

    def get_graph(self) -> Runnable:
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        # Define the edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()
