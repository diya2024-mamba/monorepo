import logging
import os
import pickle

from kiwipiepy import Kiwi
from langchain.schema import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from .metadata import MetadataVectorStore

logger = logging.getLogger(__name__)

kiwi = Kiwi()


def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]


class BM25VectorStore(MetadataVectorStore):
    path = "db/bm25.pkl"

    def preprocess(self) -> None:
        logger.info("Preprocessing data for BM25VectorStore")

        retriever = BM25Retriever.from_documents(
            documents=self._preprocess_metadata(),
            preprocess_func=kiwi_tokenize,
        )
        with open(self.path, "wb") as f:
            pickle.dump(retriever, f)

        logging.info(f"Saved vectorstore to {self.path}")

    def as_retriever(self) -> BaseRetriever:
        if not os.path.exists(self.path):
            self.preprocess()
        assert os.path.exists(self.path), f"{self.path} does not exist"

        with open(self.path, "rb") as f:
            retriever = pickle.load(f)

        return retriever
