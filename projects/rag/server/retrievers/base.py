import logging
import os
from abc import ABC, abstractmethod

from langchain.schema import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    path: str

    @abstractmethod
    def preprocess(self) -> None:
        pass

    def as_retriever(self) -> BaseRetriever:
        if not os.path.exists(self.path):
            self.preprocess()
        assert os.path.exists(self.path), f"{self.path} does not exist"

        vectorstore = FAISS.load_local(
            folder_path=self.path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever()


class TextChunkVectorStore(VectorStore):
    path = "db/text-chunk"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

    def preprocess(self) -> None:
        logger.info("Preprocessing data for TextChunkVectorStore")

        path = "data/ko_script3.txt"
        assert os.path.exists(path), f"{path} does not exist"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            is_separator_regex=False,
            **self.kwargs,
        )
        with open(path) as f:
            text = f.read()
        docs = text_splitter.create_documents([text])

        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
        )
        vectorstore.save_local(self.path)

        logging.info(f"Saved vectorstore to {self.path}")
