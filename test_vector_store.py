
from langchain_core.documents import Document

from src.vector_store import VectorStore

docs = [
    Document(
        page_content="Apple is a fruit",
        metadata={"source": "file1"}
    ),
    Document(
        page_content="Banana is yellow",
        metadata={"source": "file1"}
    )
]

import numpy as np

embeddings = [
    np.random.rand(1536),
    np.random.rand(1536)
]

store = VectorStore()

file_hash = "file1_hash_v1"
result = store.collection.get(
    include=["documents", "metadatas"]
)

print(len(result["ids"]))
print(result["metadatas"][:2])