""" 检索器
"""

import json
import copy
from loguru import logger

from .emb_searcher import EmbSearcher

from embedding import BaseEmbeddings

class Searcher:
    def __init__(self, emb_model: BaseEmbeddings, vec_db_path: str) -> None:
        self.emb_model = emb_model
        self.emb_searcher = EmbSearcher()
        self.emb_searcher.load(vec_db_path)

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



