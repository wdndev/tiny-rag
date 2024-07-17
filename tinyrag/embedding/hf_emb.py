import torch
from typing import Dict, List, Optional, Tuple, Union
from .base_emb import BaseEmbeddings
from sentence_transformers import SentenceTransformer

class HFSTEmbedding(BaseEmbeddings):
    """
    class for Hugging face sentence embeddings
    """
    def __init__(self, path: str, is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self.st_model = SentenceTransformer(path)
        self.name = "hf_model"

    def get_embedding(self, text: str) -> List[float]:
        st_embedding = self.st_model.encode([text], normalize_embeddings=True)
        return st_embedding[0].tolist()
    

