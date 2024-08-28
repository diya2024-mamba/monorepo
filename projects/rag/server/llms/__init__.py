from langchain_openai import ChatOpenAI
from llms.solar import Solar

__all__ = ["GPT4o", "Solar"]

GPT4o = ChatOpenAI(model="gpt-4o")
