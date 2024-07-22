from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import os
import re
import fitz
from nltk.tokenize import sent_tokenize

from .base_parser import BaseParser


class PDFParser(BaseParser):
    """
    Parser for PDF files
    """
    type = 'pdf'
    def __init__(self, file_path: str=None, model=None) -> None:
        super().__init__(file_path, model)
        
    def parse(self) -> List[Dict]:
        page_sents = self._to_sentences()
        if not page_sents:
            return None
        
        self.parse_output = []
        for pageno, sent in page_sents:
            file_dict = {}
            file_dict['title'] = self.metadata["title"]
            file_dict['author'] = self.metadata["author"]
            file_dict['page'] = pageno
            file_dict['content'] = sent
            file_dict['embedding'] = self.get_embedding(sent)
            file_dict['file_path'] = self.file_path
            file_dict['subject'] = self.metadata["subject"]

            self.parse_output.append(file_dict)

        return self.parse_output

    def _to_sentences(self) -> List[Tuple[int, str]]:
        """
        Parse PDF file to text [(pageno, sentence)]
        """
        if not self._check_format():
            self.parse_output = None
            return []

        pdf_doc: fitz.Document = fitz.open(self.file_path)
        
        raw_text: str = ""
        page_text: List(int, str) = []
        for pageno, page in enumerate(pdf_doc):
            page: fitz.Page
            raw_text = page.get_text("text")
            # remove hyphens 
            raw_text = re.sub(r"-\n(\w+)", r"\1", raw_text)
            raw_text = raw_text.replace("\n", " ")
            page_text.append((pageno + 1, raw_text))

        page_sents = []
        ref_flag = False
        for pageno, text in page_text:
            sents = sent_tokenize(text)
            for sent in sents:
                # remove references
                if sent.strip().lower() == "references" or sent.strip().lower().startswith("references "):
                    # print("aaa")
                    ref_flag = True
                    break
                page_sents.append((pageno, sent))
            if ref_flag:
                    break
            
        page_sents = self._merge_sentences(page_sents=page_sents)
        return page_sents
    
    def _merge_sentences(self, page_sents, len_thres=300) -> List[Tuple[int, str]]:
        """
        Merge sentences to make one sentence around 500-word length
        """
        merged_sents = []
        cur_pageno = None
        cur_sent = ''
        for pageno, sent in page_sents:
            if not cur_pageno:
                cur_pageno = pageno
                cur_sent = sent
            elif cur_pageno == pageno:
                if len(cur_sent.split()) + len(sent.split()) < len_thres:
                    cur_sent += " " + sent
                else:
                    merged_sents.append((cur_pageno, cur_sent))
                    cur_pageno = pageno
                    cur_sent = sent
            else:
                merged_sents.append((cur_pageno, cur_sent))
                cur_pageno = pageno
                cur_sent = sent
        
        if cur_sent:
            merged_sents.append((cur_pageno, cur_sent))
        return merged_sents

    
    @property
    def metadata(self) -> defaultdict:
        if not self._metadata:
            metadata = defaultdict(str)
            pdf_doc: fitz.Document = fitz.open(self.file_path)

            for k in pdf_doc.metadata.keys():
                metadata[k] = pdf_doc.metadata[k]
            
            self._metadata = metadata
        
        return self._metadata


    def _check_format(self) -> bool:
        f_path: Path = Path(self.file_path)
        return f_path.exists() and f_path.suffix == '.pdf'


if __name__ == "__main__":
    parser = PDFParser(sys.argv[1], None)
    parser._to_sentences()
    # print(parser.parse_output)
    parser.parse()