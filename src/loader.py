from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingsManager:
  def __init__(self,model_name:str='all-MiniLM-L6-v2'):
    self._model_name=model_name
    self._model=None
    self._load_model()

  def _load_model(self):
    try:
      print(f"Loading embedding model: {self._model_name}")
      self._model=SentenceTransformer(self._model_name)
    except Exception as e:
      print(f"Failed to load embedding model: {self._model_name}")
      raise e

  def generate_embeddings(self,texts:List[str]) ->np.ndarray:
    if not self._model:
      raise ValueError("Embedding Model Not Loaded")
    print(f"Generating Embeddings for {len(texts)} texts...")
    embeddings=self._model.encode(texts)
    print(f"Generated Embeddings with shape: {embeddings.shape}")
    return embeddings
