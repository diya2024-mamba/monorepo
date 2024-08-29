from langchain_openai import ChatOpenAI
from llms.llama import Llama

__all__ = ["GPT4o", "Llama3_1"]

GPT4o = ChatOpenAI(model="gpt-4o")
Llama3_1 = Llama()
