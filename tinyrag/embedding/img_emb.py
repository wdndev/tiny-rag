import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from tinyrag.embedding.base_emb import BaseEmbedding
from sentence_transformers import SentenceTransformer

class ImgEmbedding(BaseEmbedding):
    """
    class for Hugging face Image embeddings
    """
    def __init__(self, path: str, is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self.st_model = SentenceTransformer(path)
        self.name = "hf_model"

    def get_embedding(self, img: Any) -> List[float]:
        img_embedding = self.st_model.encode(img)
        return img_embedding.tolist()