from typing import List, Dict, Any

from src.llm import get_llm
from src.vector_store import VectorStore
from src.loader import EmbeddingsManager


class RAGRetriever:
    def __init__(self,
    vector_store: VectorStore,embeddings_manager:EmbeddingsManager,
    top_k: int = 5,
    score_threshold: float = 0.0,
    ):
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.llm = get_llm()

    def _retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.
        """

        top_k = top_k if top_k is not None else self.top_k
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.score_threshold
        )

        try:
            print(f"Retrieving documents for query: {query}")

            query_embedding = self.embeddings_manager.generate_embeddings([query])[0]

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            if not results.get("documents"):
                return []

            retrieved_docs = []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for rank, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            ):
                # Assumes cosine distance
                similarity_score = 1 - distance

                if similarity_score < score_threshold:
                    continue

                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata or {},
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "rank": rank,
                    }
                )

            print(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def _build_context(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Convert retrieved documents into context for the LLM.
        """

        return "\n\n".join(
            f"Source: {doc['metadata']}\n{doc['content']}"
            for doc in retrieved_docs
        )

    def _extract_sources(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Extract unique sources from metadata.
        """

        sources = []

        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})

            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or doc["id"]
            )

            if source not in sources:
                sources.append(source)

        return sources

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Retrieve context and generate a response.
        """

        try:
            retrieved_docs = self._retrieve(query=query)

            if not retrieved_docs:
                return {
                    "answer": (
                        "I couldn't find that information "
                        "in the knowledge base."
                    ),
                    "sources": [],
                    "used_rag": False,
                }

            context = self._build_context(retrieved_docs)

            prompt = f"""
            You are a helpful RAG assistant.

            Answer ONLY using the provided context.

            If the answer is not present in the context, respond exactly with:

            "I couldn't find that information in the knowledge base."

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

            response = self.llm.invoke(prompt)

            return {
                "answer": response.content,
                "sources": self._extract_sources(retrieved_docs),
                "used_rag": True,
            }

        except Exception as e:
            print(f"Error generating response: {e}")

            return {
                "answer": "An error occurred while generating the response.",
                "sources": [],
                "used_rag": False,
            }