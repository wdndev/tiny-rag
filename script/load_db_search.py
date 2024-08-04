import sys
sys.path.append(".")

import os
import json
import random
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tinyrag.utils import *

from tinyrag import SentenceSplitter
from tinyrag import Searcher


def main():
    base_dir = "data/wiki_db"
    emb_model_id = "models/bge-small-zh-v1.5"
    ranker_model_id = "models/bge-reranker-base"
    device = "cpu"
    
    searcher = Searcher(emb_model_id=emb_model_id, ranker_model_id=ranker_model_id, device=device, base_dir=base_dir)
    logger.info("search init success!")
    searcher.load_db()
    logger.info("search load database success!")

    query = "机器学习是人工智能(AI) 和计算机科学的一个分支,专注于使用数据和算法,模仿人类学习的方式,逐步提高自身的准确性。"

    result_list = searcher.search(query, 3)

    for text in result_list:
        print(text)




if __name__ == "__main__":
    main()
