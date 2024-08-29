from langchain_core.prompts import ChatPromptTemplate
from llms import Llama3_1
from rich import print


def test_call():
    llm = Llama3_1
    prompt = ChatPromptTemplate.from_messages(
        [("system", "you are a bot"), ("human", "{input}")]
    )
    chain = prompt | llm
    output = chain.invoke("hello")
    print(output)
