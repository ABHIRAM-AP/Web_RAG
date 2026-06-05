from vector_store import VectorStore
from loader import EmbeddingsManager
from typing import List,Dict,Any

class RAGRetriever:

    def __init__(self,vector_store:VectorStore,embeddings_manager:EmbeddingsManager):
        self.vector_store=vector_store
        self.embeddings_manager=embeddings_manager

    def retrieve(self,query:str,top_k:int=5,score_threshold:float=0.0)->List[Dict[str,Any]]:
        print(f"Retrieving documents for query:{query}")
        print(f"Top K:{top_k},score_threshold:{score_threshold}")

        query_embedding=self.embeddings_manager.generate_embeddings([query])[0]

        try:
            results=self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
            
            retrieved_docs=[]
            if results['documents'] and results['documents'][0]:
                documents=results['documents'][0]
                metadatas=results['metadatas'][0]
                distances=results['distances'][0]
                ids=results['ids'][0]

                for i, (doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    similarity_score = 1-distance

                    if similarity_score >=score_threshold:
                        retrieved_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'similarity_score':similarity_score,
                            'distance':distance,
                            'rank':i+1,
                        })
                        print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
                    else:
                        print("No relevant documents...")

                return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval {e}")
            return []