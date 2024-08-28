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

        documents = self.retriever.invoke(user_question, character=user_character)
        self.logger.debug("Retrieved documents: %s", documents)

        state["documents"] = documents
        return state

    def generate(self, state: GraphState, temperature: float = None):
        self.logger.debug("---GENERATE---")
        user_character = state["user_character"]
        ai_character = state["ai_character"]
        user_question = state["user_question"]
        documents = state["documents"]
        if temperature is not None:
            self.llm.temperature = temperature

        system_prompt = """
            You are an excellent mimic. You are very skilled at speaking like a real character based on the given script.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "<<<character>>> \n\n {ai_character} \n\n <<<script>>> \n\n {document} \n\n <<<characteristics>>> \n\n \
                1. Respond to {user_character} based on the script. \n 2. Conduct the conversation in one turn each. \n 3. Speak like the character. \n 4. Speak Korean. \n\n\
                <<<Conversation>>> {question}",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        generation = chain.invoke(
            {
                "user_character": user_character,
                "ai_character": ai_character,
                "document": documents,
                "question": user_question,
            }
        )
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
