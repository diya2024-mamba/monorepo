import logging
import os
from typing import Any, Optional

from langchain.schema import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .base import VectorStore

logger = logging.getLogger(__name__)


class MetadataRetriever(BaseRetriever):
    vectorstore: FAISS
    retriever: Optional[BaseRetriever]
    script: dict[str, Any] = {}
    front_script: int = 1
    back_script: int = 3

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = self.vectorstore.as_retriever()
        self.script = {
            doc.metadata["script_id"]: {
                "character": doc.metadata["character"],
                "content": doc.page_content,
            }
            for doc in self.vectorstore.docstore._dict.values()
        }

    def _check_character(self, documents: list[Document], character: str) -> list:
        doc_index = []
        for idx, doc in enumerate(documents):
            script_id = int(doc.metadata["script_id"])

            # 대본 뒷부분 캐릭터 검색
            for i in range(1, self.back_script + 1):
                if len(self.script) < script_id + i:
                    break
                elif self.script[str(script_id + i)]["character"] == character:
                    doc_index.append(idx)
                    break
        return [documents[i] for i in doc_index]

    def _make_script(self, script_id: int) -> Document:
        new_script = ""
        for i in range(self.front_script, 0, -1):
            if script_id - i < 0:
                continue
            new_script += (
                self.script[str(script_id - i)]["character"]
                + " : "
                + self.script[str(script_id - i)]["content"]
                + "\n"
            )

        for i in range(self.back_script + 1):
            if len(self.script) < script_id + i:
                break
            new_script += (
                self.script[str(script_id + i)]["character"]
                + " : "
                + self.script[str(script_id + i)]["content"]
                + "\n"
            )

        return Document(page_content=new_script)

    def _get_relevant_documents(self, query: str, *, character: str) -> list[Document]:
        documents = self.retriever.invoke(query)
        new_documents = self._check_character(documents, character)
        if new_documents:
            documents = new_documents

        output = []
        for doc in documents:
            script_id = int(doc.metadata["script_id"])
            output.append(self._make_script(script_id))

        return output


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

    def as_retriever(self) -> BaseRetriever:
        if not os.path.exists(self.path):
            self.preprocess()
        assert os.path.exists(self.path), f"{self.path} does not exist"

        vectorstore = FAISS.load_local(
            folder_path=self.path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        return MetadataRetriever(vectorstore=vectorstore)
