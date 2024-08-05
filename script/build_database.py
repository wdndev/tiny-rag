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


def process_text(item, sent_split_model):
    completion = item["completion"]
    sent_res = sent_split_model.split_text(completion)
    return sent_res



def main():
    json_path = "data/raw_data/wikipedia-cn-20230720-filtered.json"
    raw_data_list = read_json_to_list(json_path)
    logger.info("load raw data success! ")
    # 数据太多了，随机采样 5w 条数据
    raw_data_part = random.sample(raw_data_list, 50000)

    print(len(raw_data_part))
    print(raw_data_part[10])

    sent_split_model_id = "models/nlp_bert_document-segmentation_chinese-base"
    sent_split_model = SentenceSplitter(use_model=False, sentence_size=256, model_path=sent_split_model_id)
    logger.info("load sentence splitter model success! ")

    # logger.info("split sentence ...... ")
    # txt_list = []
    # for item in tqdm(raw_data_part):
    #     completion = item["completion"]
    #     sent_res = sent_split_model.split_text(completion)
    #     txt_list.extend(sent_res)

    txt_list = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_item = {executor.submit(process_text, item, sent_split_model): item for item in raw_data_part}
        
        for future in tqdm(as_completed(future_to_item), total=len(raw_data_part)):
            try:
                sent_res = future.result()
                sent_res = [item for item in sent_res if len(item) > 5]
                txt_list.extend(sent_res)
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")

    jsonl_list = [{"text": item} for item in txt_list]
    write_list_to_jsonl(jsonl_list, "data/wiki_db/split_sentence.jsonl")

    logger.info("split sentence success, all sentence number: ", len(txt_list))

    base_dir = "data/wiki_db"
    emb_model_id = "models/bge-small-zh-v1.5"
    ranker_model_id = "models/bge-reranker-base"
    device = "cpu"

    
    searcher = Searcher(emb_model_id=emb_model_id, ranker_model_id=ranker_model_id, device=device, base_dir=base_dir)
    logger.info("load search model success!")
    logger.info("build database ...... ")
    searcher.build_db(txt_list)
    logger.info("build database success, starting save .... ")
    searcher.save_db()
    logger.info("save database success!  ")




if __name__ == "__main__":
    main()
