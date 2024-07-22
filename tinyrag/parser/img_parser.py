from pathlib import Path
from typing import List, Dict
import sys
import os
from PIL import Image

from .base_parser import BaseParser


class ImgParser(BaseParser):
    """
    Parser for image files
    """
    type = 'image'
    def __init__(self, file_path: str=None, model=None) -> None:
        super().__init__(file_path, model)
        
    def parse(self) -> List[Dict]:

        img = Image.open(self.file_path)
        
        self.parse_output = []
        file_dict = {}
        file_dict['content'] = None
        file_dict['embedding'] = self.get_embedding(img)
        # print("embedding: ", type(file_dict['embedding']), len(file_dict['embedding']))
        file_dict['file_path'] = self.file_path
            
        self.parse_output.append(file_dict)
        
        return self.parse_output


    def _check_format(self) -> bool:
        f_path: Path = Path(self.file_path)
        return f_path.exists() and f_path.suffix in ['png', 'jpg', 'jpeg']