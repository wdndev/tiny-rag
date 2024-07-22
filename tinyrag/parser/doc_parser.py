# pip install python-docx

from pathlib import Path
from collections import defaultdict  
from docx import Document  
from typing import List, Dict, Tuple  
from nltk.tokenize import sent_tokenize  

from .base_parser import BaseParser

class WordParser(BaseParser):  
    """  
    Parser for Word files (.docx)  
    """  
    type = 'docx'  
      
    def __init__(self, file_path: str=None, model=None) -> None:  
        super().__init__(file_path, model)  
  
    def parse(self) -> List[Dict]:  
        page_sents = self._to_sentences()  
        if not page_sents:  
            return None  
  
        self.parse_output = []  
        for pageno, sents in page_sents:  
            for sent in sents:  
                file_dict = {}  
                file_dict['title'] = None
                file_dict['author'] = None
                file_dict['page'] = pageno  
                file_dict['content'] = sent  
                file_dict['embedding'] = self.get_embedding(sent) 
                file_dict['file_path'] = self.file_path  
                file_dict['subject'] = None
  
                self.parse_output.append(file_dict)  
  
        return self.parse_output  
  
    def _to_sentences(self) -> List[Tuple[int, str]]:  
        """  
        Parse Word file to text [(pageno, sentence)]  
        """  
        if not self._check_format():  
            self.parse_output = None  
            return []  
  
        word_doc = Document(self.file_path)  
  
        raw_text = ""  
        page_text = []  
        pageno = 1  
        for para in word_doc.paragraphs:  
            raw_text = para.text  
            raw_text = raw_text.replace("\n", " ")  
            page_text.append((pageno, raw_text))  
  
        page_sents = map(lambda x: (x[0], sent_tokenize(x[1])), page_text)  
        return page_sents  
  
    @property  
    def metadata(self) -> defaultdict:  
        if not self._metadata:  
            # Word files don't have built-in metadata like PDFs, so we set it to empty strings
            metadata = defaultdict(str)  
            self._metadata = metadata  
  
        return self._metadata  
  
    def _check_format(self) -> bool:  
        f_path = Path(self.file_path)  
        return f_path.exists() and f_path.suffix == '.docx'  