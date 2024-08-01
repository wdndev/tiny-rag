from collections import defaultdict
from typing import List, Any, Optional, Dict

from tinyrag.embedding.base_emb import BaseEmbedding

class BaseParser:
    """
    Top class of data parser
    """
    type = None
    def __init__(self, file_path: str, model: BaseEmbedding = None) -> None:
        self.file_path: str = file_path
        self.model: BaseEmbedding = model
        self._metadata: Optional[defaultdict] = None
        self.parse_output: Any = None


    def parse(self) -> List[Dict]:
        raise NotImplementedError()

    # def _to_sentences(self) -> List[Any]:
    #     """
    #     Parse file to sentences
    #     """
    #     raise NotImplementedError()
    
    def _check_format(self) -> bool:
        """
        Check input file format
        """
        raise NotImplementedError()
    
    @property
    def metadata(self) -> defaultdict:
        """
        Parse metadata
        """
        raise NotImplementedError()
    
    def get_embedding(self, obj: Any):
        if self.model is not None:
            return self.model.get_embedding(obj)
        else:
            return None