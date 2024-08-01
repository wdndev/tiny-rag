import os
from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

import numpy as np

class BaseEmbedding(ABC):
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
        self.name = ""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
    @classmethod
    def cosine_similarity2(cls, vector1: List[float], vector2: List[float]) -> float:
        sim = F.cosine_similarity(torch.Tensor(vector1), torch.Tensor(vector2), dim=-1)
        return sim.numpy().tolist()