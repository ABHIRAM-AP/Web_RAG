from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from src.retriever import RAGRetriever
from src.loader import EmbeddingsManager
from src.vector_store import VectorStore
from src.utils.hash_generator import generate_content_hash

# URLS = [
# "https://docs.langchain.com/langsmith/evaluation-approaches"
# ]
URLS=input("Enter a url to feed:")
loader = WebBaseLoader(
    web_paths=[URLS],
    requests_per_second=2,
    raise_for_status=False,
)
docs=loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
embedding_manager=EmbeddingsManager()
vectorstore=VectorStore()


for doc in docs:
    file_hash = generate_content_hash(
        doc.metadata["source"]
    )

    chunks = splitter.split_documents([doc])

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)

    vectorstore.incremental_add_docs(
        chunks,
        embeddings,
        file_hash
    )

question = "Summarize the RAG page of Google Cloud in 3 bullet points using only the indexed source."
rag_retriever=RAGRetriever(vector_store=vectorstore,embeddings_manager=embedding_manager)
final_answer = rag_retriever.generate_response(question)

print(final_answer)
