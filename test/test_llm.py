import sys
sys.path.append(".")

from tinyrag import Qwen2LLM
from tinyrag import TinyLLM



def qwen2_test():
    model_id = "models/Qwen2-0_5B"
    
    qwen2_llm = Qwen2LLM(model_id_key=model_id, device="cpu")
    res = qwen2_llm.generate("请介绍一下北京")

    print(len(res))
    print(type(res))
    print(res)

def tinyllm_test():
    model_id = "models/tiny_llm_sft_92m"
    tiny_llm = TinyLLM(model_id_key=model_id, device="cpu")
    res = tiny_llm.generate("请介绍一下北京")

    print(len(res))
    print(type(res))
    print(res)




if __name__ == "__main__":
    tinyllm_test() 