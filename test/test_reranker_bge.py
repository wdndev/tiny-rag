import sys
sys.path.append(".")

from tinyrag import RerankerBGEM3

def test_bge_reranker():

    model_id = "models/bge-reranker-base"
    bge_m3 = RerankerBGEM3(model_id_key=model_id, device="cpu")


    query = "机器学习是人工智能(AI) 和计算机科学的一个分支,专注于使用数据和算法,模仿人类学习的方式,逐步提高自身的准确性。"
    txt_list = [
        "昭通机场（ZPZT）是位于中国云南昭通的民用机场，始建于1935年。",
        "我的英雄学院：英雄新世纪是一部于2019年12月20日上映的日本动画电影。",
        "Python是一种广泛使用的高级编程语言，具有简单易读的语法。",
        "人工智能（AI）是计算机科学的一个分支，旨在创建智能机器。",
        "机器学习是人工智能的一个子集，主要关注从数据中学习。",
        "深度学习是机器学习的一个分支，使用神经网络进行学习。",
        "自然语言处理（NLP）涉及计算机与人类语言的交互。",
        "数据科学是一门利用科学方法、算法和系统来提取数据中的知识。",
        "统计学是通过收集、分析、解释和呈现数据来得出结论的学科。",
        "大数据技术用于处理和分析海量的数据集，具有处理速度快、存储容量大等特点。"
    ]
    
    recall_list = bge_m3.rank(query, txt_list)

    for text in recall_list:
        print(text)


if __name__ == "__main__":
    test_bge_reranker()
