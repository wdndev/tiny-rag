from pathlib import Path
from typing import Any, List
from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .doc_parser import WordParser
from .ppt_parser import PPTXParser
from .md_parser import MDParser
from .txt_parser import TXTParser
from .img_parser import ImgParser

from config import IMAGE_TYPES

import nltk
nltk.download("punkt")


parsers: List[BaseParser] = [PDFParser, WordParser, PPTXParser, MDParser, TXTParser, ImgParser]


def _get_parser(suffix: str) -> BaseParser:
    for parser in parsers:
        if parser.type.lower() == suffix.lower():
            return parser
    return None


def process_file(file_path: str, suffix: Any, model: Any):
    fpath = Path(file_path)
    suffix = suffix if suffix is not None else fpath.suffix.strip('.')
    if suffix in IMAGE_TYPES:
        suffix = "image"
    
    parser = _get_parser(suffix)
    print(parser)
    if not parser:
        raise NotImplementedError("Suffix of file is not supported.")
    
    return parser(file_path, model, None).parse()