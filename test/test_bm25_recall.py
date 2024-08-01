import sys
sys.path.append(".")

from tinyrag import BM25Retriever

import jieba
import nltk
from nltk.tokenize import word_tokenize
# 确保安装nltk数据
nltk.download('punkt')

def test_bm25_recall():
    # 示例使用
    # txt_list = [
    #     "昭通机场（ZPZT）是位于中国云南昭通的民用机场，始建于1935年。",
    #     "我的英雄学院：英雄新世纪是一部于2019年12月20日上映的日本动画电影。",
    #     "Python是一种广泛使用的高级编程语言，具有简单易读的语法。",
    #     "人工智能（AI）是计算机科学的一个分支，旨在创建智能机器。",
    #     "机器学习是人工智能的一个子集，主要关注从数据中学习。",
    #     "深度学习是机器学习的一个分支，使用神经网络进行学习。",
    #     "自然语言处理（NLP）涉及计算机与人类语言的交互。",
    #     "数据科学是一门利用科学方法、算法和系统来提取数据中的知识。",
    #     "统计学是通过收集、分析、解释和呈现数据来得出结论的学科。",
    #     "大数据技术用于处理和分析海量的数据集，具有处理速度快、存储容量大等特点。"
    # ]
    txt_list = []

    # 创建BM25Retriever实例
    bm25_retriever = BM25Retriever(txt_list)

    # # 可选地保存分词结果到文件
    # bm25_retriever.save_bm25_data()

    # 或者从保存的文件中加载分词结果
    bm25_retriever.load_bm25_data()

    query = "机器学习是人工智能(AI) 和计算机科学的一个分支,专注于使用数据和算法,模仿人类学习的方式,逐步提高自身的准确性。"
    top_n_texts = bm25_retriever.search(query, top_n=2)

    # 输出最相似的文本
    for text in top_n_texts:
        print(text)

if __name__ == "__main__":
    test_bm25_recall()



# import unittest

# class TestBM25Retriever(unittest.TestCase):
#     def setUp(self):
#         self.txt_list = [
#             "昭通机场（ZPZT）是位于中国云南昭通的民用机场，始建于1935年。",
#             "我的英雄学院：英雄新世纪是一部于2019年12月20日上映的日本动画电影。",
#             "Python是一种广泛使用的高级编程语言，具有简单易读的语法。",
#             "人工智能（AI）是计算机科学的一个分支，旨在创建智能机器。",
#             "机器学习是人工智能的一个子集，主要关注从数据中学习。",
#             "深度学习是机器学习的一个分支，使用神经网络进行学习。",
#             "自然语言处理（NLP）涉及计算机与人类语言的交互。",
#             "数据科学是一门利用科学方法、算法和系统来提取数据中的知识。",
#             "统计学是通过收集、分析、解释和呈现数据来得出结论的学科。",
#             "大数据技术用于处理和分析海量的数据集，具有处理速度快、存储容量大等特点。"
#         ]
#         self.retriever = BM25Retriever(self.txt_list)
#         self.file_path = "test_bm25_data.pkl"
#         self.retriever.save_bm25_data(self.file_path)

#     def test_save_load_data(self):
#         """测试数据保存和加载功能。"""
#         new_retriever = BM25Retriever([])
#         new_retriever.load_bm25_data(self.file_path)
#         self.assertEqual(self.retriever.data_list, new_retriever.data_list)
#         self.assertEqual(self.retriever.tokenized_corpus, new_retriever.tokenized_corpus)

#     def test_bm25_retrieval(self):
#         """测试 BM25 召回功能。"""
#         query = "英雄新世纪"
#         top_n = 3
#         top_n_texts = self.retriever.bm25_retrieval(query, top_n)
#         self.assertEqual(len(top_n_texts), top_n)
#         # 可选: 验证返回的结果是否包含查询的关键字
#         for text in top_n_texts:
#             self.assertIn(query, text)

#     def tearDown(self):
#         import os
#         try:
#             os.remove(self.file_path)
#         except FileNotFoundError:
#             pass

# if __name__ == "__main__":
#     unittest.main()


