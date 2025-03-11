import json
import logging
import os

from langchain_community.vectorstores import FAISS

from .base import VectorStore

logger = logging.getLogger(__name__)


class RelationshipSearch:
    def __init__(self, character):
        self.character = character
        self.relationship = self.relationship_map()
        self.relationship_list = list(self.relationship.keys())

    def relationship_map(self):
        path = "open_data/character_relationship.json"
        print(os.path.realpath(__file__))
        assert os.path.exists(path), f"{path} does not exist"

        with open(path, "r", encoding="utf-8") as f:
            relationship = json.load(f)
        return relationship[self.character]

    def search_relationship_document(self, name):
        if name in self.relationship:
            return self.relationship[name]
        else:
            return f"{name} is not in document"


class StoryVectorStore(VectorStore):
    path = "open_data/harry_story_db"
    db: FAISS = None

    def preprocess(self) -> None:
        pass
