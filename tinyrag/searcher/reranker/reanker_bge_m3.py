import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple

from tinyrag.searcher.reranker.reranker_base import RankerBase

class RerankerBGEM3(RankerBase):
    def __init__(self, model_id_key: str, device: str = "", is_api=False) -> None:
        super().__init__(model_id_key, is_api)
        
        self.device = torch.device(device if device else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id_key = model_id_key
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_key)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id_key)
        self.model.to(self.device)  # 将模型移动到指定设备
        self.model.eval()  # 设置模型为评估模式

    def rank(self, query: str, candidate_query, top_n=3) -> List[Tuple[float, str]]:
        # 创建查询和文本对
        pairs = [[query, txt] for txt in candidate_query]

        # 计算得分
        with torch.no_grad():  # 不计算梯度以节省内存
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        # 将得分和文本对结合，并按得分排序
        scored_query_list = list(zip(scores, candidate_query))
        scored_query_list.sort(key=lambda x: x[0], reverse=True)  # 按得分降序排列

        # 取前 top_n 的结果
        top_n_results = scored_query_list[:top_n]

        return top_n_results



