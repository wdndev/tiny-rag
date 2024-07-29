import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

class RankerBase(ABC):
    def __init__(self,  model_id_key: str, is_api: bool = False) -> None:
        super().__init__()
        
        # 设置模型标识符
        self.model_id_key = model_id_key

        self.is_api = is_api

    @abstractmethod
    def rank(self, query: str, candidate_query: List[str], top_n=3)  -> List[Tuple[float, str]]:
        # 当尝试实例化该抽象基类时抛出未实现错误
        raise NotImplementedError


