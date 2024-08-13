import sys
sys.path.append(".")

import os

from tinyrag import BaseParser
from tinyrag import PDFParser
from tinyrag import WordParser
from tinyrag import PPTXParser
from tinyrag import MDParser
from tinyrag import TXTParser
from tinyrag import ImgParser

from tinyrag import HFSTEmbedding, ImgEmbedding

from tinyrag import parser_file

def get_file_paths(directory):
    file_paths = []
    # os.walk()生成目录树中文件夹和文件的元组
    for root, dirs, files in os.walk(directory):
        for file in files:
            # os.path.join()用来拼接目录和文件名
            file_paths.append(os.path.join(root, file))
    return file_paths


def pdf_test():
    file_path = "data/parser_test/pdf/C++17.pdf"
    model_id = "models/bge-small-zh-v1.5"
    hf_emb = HFSTEmbedding(path=model_id)
    pdf_parser = PDFParser(file_path, hf_emb)
    page_sents = pdf_parser._to_sentences()
    for pageno, sent in page_sents:
        print("pageno: ", pageno)
        print("sent: ", sent)
        break

    parse_output = pdf_parser.parse()
    for file_dict in parse_output:
        print("file_dict: ", file_dict)
        break

def word_test():
    file_path = "data/parser_test/word/test1.docx"
    model_id = "models/bge-small-zh-v1.5"
    hf_emb = HFSTEmbedding(path=model_id)
    word_parser = WordParser(file_path, hf_emb)
    page_sents = word_parser._to_sentences()
    for pageno, sent in page_sents:
        print("pageno: ", pageno)
        print("sent: ", sent)
        break

    parse_output = word_parser.parse()
    for file_dict in parse_output:
        print("file_dict: ", file_dict)
        break

def ppt_test():
    file_path = "data/parser_test/ppt/test1.pptx"
    model_id = "models/bge-small-zh-v1.5"
    hf_emb = HFSTEmbedding(path=model_id)
    ppt_parser = PPTXParser(file_path, hf_emb)
    page_sents = ppt_parser._to_sentences()
    for pageno, sent in page_sents:
        print("pageno: ", pageno)
        print("sent: ", sent)
        break

    parse_output = ppt_parser.parse()
    for file_dict in parse_output:
        print("file_dict: ", file_dict)
        break

def md_test():
    file_path = "data/parser_test/md/1.处理日期和时间的chrono库.md"
    model_id = "models/bge-small-zh-v1.5"
    hf_emb = HFSTEmbedding(path=model_id)
    md_parser = MDParser(file_path, hf_emb)
    page_sents = md_parser._to_sentences()
    for pageno, sent in page_sents:
        print("pageno: ", pageno)
        print("sent: ", sent)
        break

    parse_output = md_parser.parse()
    for file_dict in parse_output:
        print("file_dict: ", file_dict)
        break

def txt_test():
    file_path = "data/parser_test/txt/c++对象与模型.txt"
    model_id = "models/bge-small-zh-v1.5"
    hf_emb = HFSTEmbedding(path=model_id)
    txt_parser = TXTParser(file_path, hf_emb)
    page_sents = txt_parser._to_sentences()
    for pageno, sent in page_sents:
        print("pageno: ", pageno)
        print("sent: ", sent)
        break

    parse_output = txt_parser.parse()
    for file_dict in parse_output:
        print("file_dict: ", file_dict)
        break

def img_test():
    file_path = "data/parser_test/img/Llama3_Repo.jpeg"
    model_id = "models/clip-ViT-B-32"
    img_emb = ImgEmbedding(path=model_id)
    img_parser = ImgParser(file_path, img_emb)

    parse_output = img_parser.parse()
    for file_dict in parse_output:
        print("file_dict: ", file_dict)
        break


if __name__ == "__main__":
    # pdf_test()
    # word_test()
    # ppt_test()
    # md_test()
    # txt_test()
    img_test()