
import os
import json
import random
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from tinyrag import BaseLLM, Qwen2LLM, TinyLLM
from tinyrag import Searcher
from tinyrag import SentenceSplitter
from tinyrag.utils import write_list_to_jsonl


RAG_PROMPT_TEMPALTE="""参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""

@dataclass
class RAGConfig:
    base_dir:str = "data/wiki_db"
    llm_model_id:str = "models/tiny_llm_sft_92m"
    emb_model_id: str = "models/bge-small-zh-v1.5"
    ranker_model_id:str = "models/bge-reranker-base"
    device:str = "cpu"
    sent_split_model_id:str = "models/nlp_bert_document-segmentation_chinese-base"
    sent_split_use_model:bool = False
    sentence_size:int = 256
    model_type: str = "tinyllm"

def process_docs_text(docs_text, sent_split_model):
    sent_res = sent_split_model.split_text(docs_text)
    return sent_res

class TinyRAG:
    def __init__(self, config:RAGConfig) -> None:
        print("config: ", config)
        self.config = config
        self.searcher = Searcher(
            emb_model_id=config.emb_model_id,
            ranker_model_id=config.ranker_model_id,
            device=config.device,
            base_dir=config.base_dir
        )

        if self.config.model_type == "qwen2":
            self.llm:BaseLLM = Qwen2LLM(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
        elif self.config.model_type == "tinyllm":
            self.llm:BaseLLM = TinyLLM(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
        else:
            raise "failed init LLM, the model type is [qwen2, tinyllm]"

    def build(self, docs: List[str]):
        """ 注意： 构建数据库需要很长时间
        """
        self.sent_split_model = SentenceSplitter(
            use_model=False, 
            sentence_size=self.config.sentence_size, 
            model_path=self.config.sent_split_model_id
        )
        logger.info("load sentence splitter model success! ")
        txt_list = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_item = {executor.submit(process_docs_text, item, self.sent_split_model): item for item in docs}
            
            for future in tqdm(as_completed(future_to_item), total=len(docs)):
                try:
                    sent_res = future.result()
                    sent_res = [item for item in sent_res if len(item) > 100]
                    txt_list.extend(sent_res)
                except Exception as exc:
                    logger.error(f"Generated an exception: {exc}")

        jsonl_list = [{"text": item} for item in txt_list]
        write_list_to_jsonl(jsonl_list, self.config.base_dir + "/split_sentence.jsonl")
        logger.info("split sentence success, all sentence number: ", len(txt_list))
        logger.info("build database ...... ")
        self.searcher.build_db(txt_list)
        logger.info("build database success, starting save .... ")
        self.searcher.save_db()
        logger.info("save database success!  ")

    def load(self):
        self.searcher.load_db()
        logger.info("search load database success!")

    def search(self, query: str, top_n:int = 3) -> str:
        # LLM的初次回答
        llm_result_txt = self.llm.generate(query)
        # 数据库检索的文本
        ## 拼接 query和LLM初次生成的结果，查找向量数据库
        search_content_list = self.searcher.search(query=query+llm_result_txt+query, top_n=top_n)
        content_list = [item[1] for item in search_content_list]
        context = "\n".join(content_list)
        # 构造 prompt
        prompt_text = RAG_PROMPT_TEMPALTE.format(
            context=context,
            question=query,
            answer=llm_result_txt
        )
        logger.info("prompt: {}".format(prompt_text))
        # 生成最终答案
        output = self.llm.generate(prompt_text)

        return output
        


