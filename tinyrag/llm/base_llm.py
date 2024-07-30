import os
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Base class for embeddings
    """
    def __init__(self, model_id_key: str, device:str = "cpu", is_api=False) -> None:
        super().__init__()
        self.model_id_key = model_id_key
        self.device = device
        self.is_api = is_api

    @abstractmethod
    def generate(self, content: str) -> str:
        raise NotImplemented
