#!/usr/bin/env python
from enum import StrEnum

from chains import base_graph
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from llms import ChatOpenAI, Solar
from pydantic import BaseModel
from retrievers import BM25VectorStore, MetadataVectorStore, TextChunkVectorStore

app = FastAPI(title="LangChain Server for RAG")


class LLM(StrEnum):
    OPENAI = "openai"
    SOLAR = "solar"


class Retriever(StrEnum):
    TEXTCHUNK = "textchunk"
    METADATA = "metadata"
    BM25 = "bm25"


class RAG(StrEnum):
    BASE = "base"


class Input(BaseModel):
    llm: LLM
    retriever: Retriever
    rag: RAG
    character: str
    prompt: str


@app.post("/invoke")
async def invoke(input: Input) -> JSONResponse:
    match input.llm:
        case LLM.OPENAI:
            llm = ChatOpenAI()
        case LLM.SOLAR:
            llm = Solar()
        case _:
            raise ValueError(f"Invalid LLM: {input.llm}")

    match input.retriever:
        case Retriever.TEXTCHUNK:
            retriever = TextChunkVectorStore().as_retriever()
        case Retriever.METADATA:
            retriever = MetadataVectorStore().as_retriever()
        case Retriever.BM25:
            retriever = BM25VectorStore().as_retriever()
        case _:
            raise ValueError(f"Invalid Retriever: {input.retriever}")

    match input.rag:
        case RAG.BASE:
            graph = base_graph(llm, retriever)
        case _:
            raise ValueError(f"Invalid RAG: {input.rag}")

    output = graph.invoke(
        {
            "user_question": input.prompt,
            "user_character": input.character,
        }
    )
    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
