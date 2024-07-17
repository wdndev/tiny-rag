from tinyrag import ZhipuEmbedding
from tinyrag import HFSTEmbedding


def zhipuai_test():
    api_key = "4f50b90c8cf1763b990ba942ddd955a8.9PHAwesEpIIfsB2a"
    zhipu_emb = ZhipuEmbedding(api_key=api_key)
    emb1 = zhipu_emb.get_embedding("你好") 
    emb2 = zhipu_emb.get_embedding("您好")

    print(emb1)
    print(zhipu_emb.cosine_similarity(emb1, emb2))
    print(zhipu_emb.cosine_similarity2(emb1, emb2))

def hf_test():
    model_id = "models/bge-small-zh-v1.5"
    hf_emb = HFSTEmbedding(path=model_id)
    emb1 = hf_emb.get_embedding("你好") 
    emb2 = hf_emb.get_embedding("您好")

    print(emb1)
    print(hf_emb.cosine_similarity(emb1, emb2))
    print(hf_emb.cosine_similarity2(emb1, emb2))


if __name__ == "__main__":
    hf_test() 