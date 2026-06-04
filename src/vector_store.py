import os
import uuid

import chromadb


class VectorStore:

    def __init__(self,collection_name:str='url_info',persist_directory:str='data/vector_store'):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._init_store()

    
    def _init_store(self):

        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)

            self.collection=self.client.get_or_create_collection(name=self.collection_name)
            
            print(f"Vector Store initialized. Collection:{self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store :{e}")
            raise e
    
    def add_documents(self, documents, embeddings):

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):

            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(f"Successfully added {len(documents)} documents")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
