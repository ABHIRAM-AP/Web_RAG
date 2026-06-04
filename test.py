from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader import EmbeddingsManager
from src.vector_store import VectorStore

URLS = [
    "https://docs.langchain.com/oss/python/langchain/rag",
    "https://reference.langchain.com/python/langchain-community/document_loaders/web_base/WebBaseLoader"
]
loader = WebBaseLoader(
    web_paths=URLS,
    requests_per_second=2,
    raise_for_status=False,
)
docs=loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks=splitter.split_documents(docs)

embedding_manager=EmbeddingsManager()
vectorstore=VectorStore()

texts=[doc.page_content for doc in chunks]

embeddings=embedding_manager.generate_embeddings(texts)




# vectorstore.add_documents(splits,embeddings)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# question = "How does WebBaseLoader work?"

# relevant_docs = retriever.invoke(question)

# context = "\n\n".join([doc.page_content for doc in relevant_docs])
# print(context[:100])
