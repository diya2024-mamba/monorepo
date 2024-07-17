from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


def script_retriever(embedding_type, embedding_model, db, query):
    if embedding_type == 'huggingface':
        None
        # embeddings = huggingface
    else:
        # embeddings = OpenAIEmbeddings(embedding_model)
        embeddings = OpenAIEmbeddings()
    
    vector_db = FAISS.load_local(db, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()
    script = retriever.invoke(query)

    return script[0].page_content

