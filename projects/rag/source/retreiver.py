from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def script_retriever(embedding_type, embedding_model, db, query):
    """Retriver를 활용하여 사용자 질문과 가장 유사한 대본을 반환한다."""
    if embedding_type == 'huggingface':
        pass
        # None
        # embeddings = huggingface
    else:
        # embeddings = OpenAIEmbeddings(embedding_model)
        embeddings = OpenAIEmbeddings()

    vector_db = FAISS.load_local(db, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()
    script = retriever.invoke(query)

    return script[0].page_content
