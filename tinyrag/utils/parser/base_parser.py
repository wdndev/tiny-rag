from collections import defaultdict
from typing import List, Any, Optional, Dict

class BaseParser:
    """
    Top class of data parser
    """
    type = None
    def __init__(self, file_path: str, model: Any, out_path: str) -> None:
        self.file_path: str = file_path
        self.out_path: str = out_path
        self.model: Any = model
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