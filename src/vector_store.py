import os
import chromadb

from src.utils.hash_generator import generate_content_hash,generate_doc_id

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
   
    def incremental_add_docs(self,documents,embeddings,file_hash:str):
        if len(documents)!=len(embeddings):
            raise ValueError('No:of documents must match embeddings')
        
        existing=self.collection.get(
            where={"file_hash":file_hash},
            include=["metadatas"]
        )
        existing_map={}

        for doc_id,meta in zip(existing.get('ids',[]),existing.get('metadatas',[])):
            existing_map[meta.get('chunk_index')]={
                'id':doc_id,
                'chunk_hash':meta.get('chunk_hash')
            }
        
        ids=[]
        docs=[]
        embeds=[]
        metas=[]

        up_ids,up_docs,up_embeds,up_metas=[],[],[],[]

        for index,(doc,embedding)in enumerate(zip(documents,embeddings)):
            chunk_text=doc.page_content
            chunk_hash=generate_content_hash(chunk_text)
            doc_id=generate_doc_id(file_hash,index)
            
            metadata=dict(doc.metadata)
            metadata.update({
                "file_hash":file_hash,
                "chunk_index":index,
                "chunk_hash":chunk_hash,
                "content_length":len(chunk_text)
            })

            if index not in existing_map:
                ids.append(doc_id)
                docs.append(chunk_text)
                embeds.append(embedding.tolist())
                metas.append(metadata)
            
            else:
                old_hash=existing_map[index]['chunk_hash']

                if old_hash!=chunk_hash:
                    up_ids.append(existing_map[index]['id'])
                    up_docs.append(chunk_text)
                    up_embeds.append(embedding.tolist())
                    up_metas.append(metadata)
        
        if ids:
            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeds,
                metadatas=metas
            )
        if up_ids:
            self.collection.update(
                ids=up_ids,
                documents=up_docs,
                embeddings=up_embeds,
                metadatas=up_metas
            )
        print("\nSummary\n")
        print(f"Added: {len(ids)}")
        print(f"Updated: {len(up_ids)}")
        print(f"Skipped: {len(documents)-len(ids)-len(up_ids)}")


    
### ========================Outdated======================================= ####
    '''def add_documents(self, documents, embeddings,file_hash:str):

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
            metadata['file_hash']=file_hash
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
            raise'''
### ============================================================================== ####