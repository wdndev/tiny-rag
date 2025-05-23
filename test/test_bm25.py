import sys
sys.path.append(".")

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from tinyrag import BM25Okapi, BM25L, BM25Plus
import re


corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

algs = [
    BM25Okapi(tokenized_corpus),
    BM25L(tokenized_corpus),
    BM25Plus(tokenized_corpus)
]


def test_corpus_loading():
    for alg in algs:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert alg.doc_len == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]

def test_alg():
    for alg in algs:
        bm25 = alg

        query = "cool weather, oh ...."
        tokenized_query = query.split(" ")

        doc_scores = bm25.get_scores(tokenized_query)
        print("doc_scores: ", doc_scores)

        doc_top_n = bm25.get_top_n(tokenized_query, corpus, n=1)
        print("doc_top_n: ", doc_top_n)
        print("-------------------------------")


if __name__ == "__main__":
    test_alg()