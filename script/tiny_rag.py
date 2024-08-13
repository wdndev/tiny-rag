import sys
sys.path.append(".")

import os
import json
import random
import argparse
from loguru import logger
from tqdm import tqdm

from tinyrag import RAGConfig, TinyRAG

from tinyrag.utils import read_json_to_list

def build_db(config_path, data_path):

    raw_data_list = read_json_to_list(data_path)
    logger.info("load raw data success! ")
    # 数据太多了，随机采样 100 条数据
    # raw_data_part = random.sample(raw_data_list, 100)

    text_list = [item["completion"] for item in raw_data_list]

    # config_path = "config/build_config.json"
    config = read_json_to_list(config_path)
    rag_config = RAGConfig(**config)
    tiny_rag = TinyRAG(config=rag_config)
    tiny_rag.build(text_list)

def query_search(config_path):
    # config_path = "config/tiny_llm_config.json"
    config = read_json_to_list(config_path)
    rag_config = RAGConfig(**config)
    tiny_rag = TinyRAG(config=rag_config)
    logger.info("tiny rag init success!")
    tiny_rag.load()
    query = "请介绍一下北京"
    output = tiny_rag.search(query, top_n=6)
    print("output: ", output)

def main():
    parser = argparse.ArgumentParser(description='Tiny RAG Argument Parser')
    parser.add_argument("-c", '--config', type=str, default="config/qwen2_config.json", help='Tiny RAG config')
    parser.add_argument("-t", '--type', type=str, default="search", help='Tiny RAG Type [build, search]')
    parser.add_argument('-p', "--path",  type=str, default="data/raw_data/wikipedia-cn-20230720-filtered.json", help='Tiny RAG data path')

    args = parser.parse_args()

    if args.type == "build":
        build_db(args.config, args.path)
    elif args.type == "search":
        query_search(args.config)

if __name__ == "__main__":
    main()

