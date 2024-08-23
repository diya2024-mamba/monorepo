import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .base import VectorStore

logger = logging.getLogger(__name__)


class MetadataVectorStore(VectorStore):
    path = "db/metadata"
    db: FAISS = None

    def _preprocess_metadata(self) -> list:
        path = "data/ko_script3.txt"
        assert os.path.exists(path), f"{path} does not exist"

        script = []
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                script.append(line)

        sample = []
        for i in range(len(script)):
            if ":" in script[i]:
                character, dialogue = script[i].split(":")
                character, dialogue = character.strip(), dialogue.strip()
                sample.append(
                    Document(
                        page_content=dialogue,
                        metadata={"script_id": str(i), "character": character},
                    )
                )

            else:
                sample.append(
                    Document(
                        page_content=script[i],
                        metadata={"script_id": str(i), "character": "Commentary"},
                    )
                )
        return sample

    def preprocess(self) -> None:
        logger.info("Preprocessing data for MetadataVectorStore")

        vectorstore = FAISS.from_documents(
            documents=self._preprocess_metadata(),
            embedding=OpenAIEmbeddings(),
        )
        vectorstore.save_local(self.path)

        logging.info(f"Saved vectorstore to {self.path}")

    def search_metadata_faiss(self, key: str, value: str) -> dict:
        if self.db is None:
            self.db = self.as_retriever()

        docs = self.db.vectorstore.docstore._dict.items()
        return dict(filter(lambda v: v[1].metadata[key] == value, docs))
