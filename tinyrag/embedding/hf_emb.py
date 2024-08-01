import torch
from typing import Dict, List, Optional, Tuple, Union
from tinyrag.embedding.base_emb import BaseEmbedding
from sentence_transformers import SentenceTransformer, util

class HFSTEmbedding(BaseEmbedding):
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
    

