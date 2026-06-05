from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import VectorStore
from src.loader import EmbeddingsManager
from typing import List,Dict,Any
import os
from dotenv import load_dotenv

load_dotenv()

class RAGRetriever:

    def __init__(self,vector_store:VectorStore,embeddings_manager:EmbeddingsManager,top_k:int=5,score_threshold:float=0.0,):
        self.vector_store=vector_store
        self.embeddings_manager=embeddings_manager


    def _retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: {query}")

        try:
            query_embedding = self.embeddings_manager.generate_embeddings([query])[0]

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs = []

            if not results.get("documents"):
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "rank": i + 1,
                    })

            print(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def generate_response(self,query:str):
        GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
        llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash',api_key=GOOGLE_API_KEY)
        results = self._retrieve(query=query)
        print('='*10)
        print(results)
        print('='*10)
        context="\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return f"No relevant information to the query provided"
        
            # Generate the answer
        prompt=f"""Use the following context to answer the questions concisely
            Context:
            {context}
            Question: {query}
            Answer:
            """

        response = llm.invoke(prompt)
        return response.content