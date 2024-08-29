#!/usr/bin/env python
import asyncio
import logging
import os
import random
import secrets
from enum import StrEnum

from chains import AdvancedRAG
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from langgraph.errors import GraphRecursionError
from llms import GPT4o, Llama3_1
from pydantic import BaseModel
from retrievers import BM25VectorStore, MetadataVectorStore, TextChunkVectorStore

app = FastAPI(title="LangChain Server for RAG")

security = HTTPBasic()

client_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client")
app.mount("/client", StaticFiles(directory=client_path), name="client")


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = secrets.compare_digest(
        credentials.password,
        os.getenv("SERVER_PASSWORD"),
    )
    if not correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.password


# frontend
@app.get("/")
def read_root(password: str = Depends(authenticate)):
    return FileResponse(os.path.join(client_path, "index.html"))


@app.get("/ab_test")
def read_ab_test(password: str = Depends(authenticate)):
    return FileResponse(os.path.join(client_path, "static", "ab_test.html"))


class LLM(StrEnum):
    OPENAI = "openai"
    LLAMA = "llama"


class Retriever(StrEnum):
    TEXTCHUNK = "textchunk"
    METADATA = "metadata"
    BM25 = "bm25"


class RAG(StrEnum):
    BASE = "base"
    CRAG = "crag"
    SRAG = "srag"


class InvokeInput(BaseModel):
    llm: LLM
    retriever: Retriever
    rag: RAG
    user_character: str
    ai_character: str
    prompt: str


@app.post("/invoke")
async def invoke(input: InvokeInput) -> JSONResponse:
    match input.llm:
        case LLM.OPENAI:
            llm = GPT4o
        case LLM.LLAMA:
            llm = Llama3_1
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

    if input.rag not in RAG.__members__.values():
        raise ValueError(f"Invalid RAG: {input.rag}")

    rag = AdvancedRAG(retriever, llm, input.rag)
    graph = rag.get_graph()

    query = {
        "user_question": input.prompt,
        "user_character": input.user_character,
        "ai_character": input.ai_character,
    }

    try:
        output = graph.invoke(query)
    except GraphRecursionError:
        output = "Agent stopped due to max iterations."
    return output


class RandomInput(BaseModel):
    character: str
    prompt: str


def random_config() -> tuple[LLM, Retriever, RAG]:
    llm = random.choice(list(LLM))
    retriever = random.choice(list(Retriever))
    rag = random.choice(list(RAG))
    return llm, retriever, rag


@app.post("/random")
async def random_invoke(input: RandomInput) -> JSONResponse:
    config1 = random_config()
    config2 = random_config()
    while config1 == config2:
        config2 = random_config()

    input1 = InvokeInput(
        llm=config1[0],
        retriever=config1[1],
        rag=config1[2],
        ai_character=input.character,
        user_character="마지",
        prompt=input.prompt,
    )
    input2 = InvokeInput(
        llm=config2[0],
        retriever=config2[1],
        rag=config2[2],
        ai_character=input.character,
        user_character="마지",
        prompt=input.prompt,
    )

    outputs = await asyncio.gather(
        invoke(input1),
        invoke(input2),
    )
    return {
        "A": {
            "llm": config1[0],
            "retriever": config1[1],
            "rag": config1[2],
            "output": outputs[0],
        },
        "B": {
            "llm": config2[0],
            "retriever": config2[1],
            "rag": config2[2],
            "output": outputs[1],
        },
    }


class VoteConfig(BaseModel):
    llm: LLM
    retriever: Retriever
    rag: RAG


class VoteInput(BaseModel):
    winner: VoteConfig
    loser: VoteConfig


# logger for votes
vote_logger = logging.getLogger("vote_logger")
vote_logger.setLevel(logging.INFO)
vote_logger.propagate = False

file_handler = logging.FileHandler("votes.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[VOTE] %(asctime)s - %(message)s")
file_handler.setFormatter(formatter)
vote_logger.addHandler(file_handler)


@app.post("/vote")
async def vote(input: VoteInput) -> JSONResponse:
    vote_logger.info(input.json())
    return {"message": "투표가 완료되었습니다."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
