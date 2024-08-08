import json
import pickle

from preprocess import preprocess_metadata

from kiwipiepy import Kiwi
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


def kiwi_tokenize(text):
    '''형태소 분석기로 토크나이징'''        
    kiwi = Kiwi()
    return [token.form for token in kiwi.tokenize(text)]


def kiwi_bm25_db(docs, save_path=None):
    '''bm25 vector store'''
    kiwi_bm25 = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize)
    print(1)
    if save_path != None:
        with open(save_path, 'wb') as f:
            pickle.dump(kiwi_bm25, f)
    return kiwi_bm25


def search_metadata_faiss(db, key, value):
    '''faiss db에서 metadata의 key로 value 검색'''
    return dict(filter(lambda v : v[1].metadata[key]==value, db.docstore._dict.items()))


def search_metadata_json(script_id, json_path):
    '''json에서 metadata의 script_id로  검색'''
    with open(json_path, 'r', encoding='utf-8') as f:
        script_json = json.load(f)

    return script_json[str(script_id)]

if __name__ == '__main__':
    doc = preprocess_metadata('../data/ko_script.txt')
    # faiss
    faiss_db = FAISS.from_documents(doc, OpenAIEmbeddings(model="text-embedding-3-large"))
    faiss_db.save_local("../db/ko_faiss_base")
    

    
