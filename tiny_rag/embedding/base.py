import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class BaseEmbeddings:
    """ Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, texts: List[str]) -> List[float]:
        raise NotImplemented
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """ calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magntude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magntude:
            return 0
        return dot_product / magntude