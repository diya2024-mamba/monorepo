from langchain_core.prompts import ChatPromptTemplate
from llms import Solar
from rich import print


def test_call():
    llm = Solar()
    prompt = ChatPromptTemplate.from_messages(
        [("system", "you are a bot"), ("human", "{input}")]
    )
    chain = prompt | llm
    output = chain.invoke("hello")
    print(output)
