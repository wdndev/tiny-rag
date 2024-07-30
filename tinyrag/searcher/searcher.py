import json
import copy
from loguru import logger

from typing import Dict, List, Optional, Tuple, Union

from tinyrag.embedding.hf_emb import HFSTEmbedding
from tinyrag.searcher.bm25_recall.bm25_retriever import BM25Retriever
from tinyrag.searcher.emb_recall.emb_retriever import EmbRetriever
from tinyrag.searcher.reranker.reanker_bge_m3 import RerankerBGEM3

class Searcher:
    def __init__(self, emb_model_id, ranker_model_id, device, base_dir) -> None:
        # self.base_dir = "data/db"
        # emb_model_id = "models/bge-small-zh-v1.5"
        # ranker_model_id = "models/bge-reranker-base"
        # device = "cpu"
        self.base_dir = base_dir
        emb_model_id = emb_model_id
        ranker_model_id = ranker_model_id
        device = device

        # 召回
        self.bm25_retriever = BM25Retriever(base_dir=self.base_dir+"/bm_corpus")
        self.emb_model = HFSTEmbedding(path = emb_model_id)
        index_dim = len(self.emb_model.get_embedding("test_dim"))
        self.emb_retriever = EmbRetriever(index_dim=index_dim, base_dir=self.base_dir+"/faiss_idx")

        # 排序
        self.ranker = RerankerBGEM3(model_id_key = ranker_model_id, device=device)

        logger.info("Searcher init build success...")


    def build_db(self, docs: List[str]):
        self.bm25_retriever.build(docs)
        logger.info("bm25 retriever build success...")
        for doc in docs:
            doc_emb = self.emb_model.get_embedding(doc)
            self.emb_retriever.insert(doc_emb, doc)
        logger.info("emb retriever build success...")

    def save_db(self):
        # self.base_dir = base_dir
        self.bm25_retriever.save_bm25_data()
        logger.info("bm25 retriever save success...")
        self.emb_retriever.save()
        logger.info("emb retriever save success...")

    def load_db(self):
        # self.base_dir = base_dir
        self.bm25_retriever.load_bm25_data()
        logger.info("bm25 retriever load success...")
        self.emb_retriever.load()
        logger.info("emb retriever load success...")

    def search(self, query:str, top_n=3) -> list:
        bm25_recall_list = self.bm25_retriever.search(query, top_n)
        logger.info("bm25 recall text num: {}".format(len(bm25_recall_list)))
        query_emb = self.emb_model.get_embedding(query)
        emb_recall_list = self.emb_retriever.search(query_emb, top_n)
        logger.info("emb recall text num: {}".format(len(emb_recall_list)))

        recall_unique_text = set()
        for idx, text, score in bm25_recall_list:
            recall_unique_text.add(text)

        for idx, text, score in emb_recall_list:
            recall_unique_text.add(text)

        logger.info("unique recall text num: {}".format(len(recall_unique_text)))

        rerank_result = self.ranker.rank(query, list(recall_unique_text), top_n)

        return rerank_result