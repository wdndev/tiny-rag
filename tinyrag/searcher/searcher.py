""" 检索器
"""

import json
import copy
from loguru import logger

from typing import Dict, List, Optional, Tuple, Union

from embedding import BaseEmbeddings
from .emb_recall.emb_searcher import EmbSearcher
class Searcher:
    def __init__(self, emb_model: BaseEmbeddings, base_dir="data/index") -> None:
        self.emb_model = emb_model
        self.emb_searcher = EmbSearcher(base_dir)

    def build_emb_db(self, docs: List[str], index_name="index_test"):
        index_dim = len(self.emb_model.get_embedding("test_dim"))
        self.emb_searcher.build(index_dim, index_name)
        for doc in docs:
            doc_emb = self.emb_model.get_embedding(doc)
            self.emb_searcher.insert(doc_emb, doc)

    def load_emb_db(self, index_name: str):
        self.emb_searcher.load(index_name)
    
    def save_emb_db(self, index_name: str):
        self.emb_searcher.save(index_name)

    def rank(self, query, recall_list):
        rank_result = []
        for idx in range(len(recall_list)):
            new_sim = self.emb_model.cosine_similarity(query, recall_list[idx][1][0])
            rank_item = copy.deepcopy(recall_list[idx])
            rank_item.append(new_sim)
            rank_result.append(copy.deepcopy(rank_item))
        rank_result.sort(key=lambda x:x[3], reverse=True)
        return rank_result
    
    def search(self, query, nums=3):
        q_emb = self.emb_model.get_embedding(query)
        recall_result = self.emb_searcher.search(q_emb, nums)
        rank_result = self.rank(q_emb, recall_result)

        return rank_result



