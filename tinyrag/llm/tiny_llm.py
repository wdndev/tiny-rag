import os
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from tinyrag.llm.base_llm import BaseLLM

class TinyLLM(BaseLLM):
    def __init__(self, model_id_key: str, device: str = "cpu", is_api=False) -> None:
        super().__init__(model_id_key, device, is_api)

        # 从预训练模型加载因果语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_key,  # 模型标识符
            torch_dtype="auto",  # 自动选择张量类型
            device_map=self.device,  # 分布到特定设备上
            trust_remote_code=True  # 允许加载远程代码
        )
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_key,  # 分词器标识符
            use_fast=False,
            trust_remote_code=True
        )
        
        # 加载配置文件
        self.config = AutoConfig.from_pretrained(
            self.model_id_key,  # 配置文件标识符
            trust_remote_code=True  # 允许加载远程代码
        )

        if self.device == "cpu":
            self.model.float()
        
        # 设置模型为评估模式
        self.model.eval()

    def generate(self, content: str) -> str:
        sys_text = "你是由wdndev开发的个人助手。"
        input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                            "<|user|>", content.strip(), 
                            "<|assistant|>"]).strip() + "\n"
        model_inputs = self.tokenizer(input_txt, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=200
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

