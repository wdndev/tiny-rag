
import math
import numpy as np
from multiprocessing import Pool, cpu_count

class BM25:
    """ 每种算法都有其适用场景，选择哪种算法取决于具体的应用需求和数据特性。
        - 如果数据集中存在很多低频词语，那么BM25Okapi可能更适合；
        - 而对于文档长度差异较大的数据集，BM25L或BM25Plus可能表现更好。
    """
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0        # 文档总数
        self.avgdl = 0              # 文档平均长度
        self.doc_freqs = []         # 每个文档中词语的频率
        self.idf = {}               # 逆文档频率（IDF）
        self.doc_len = []           # 每个文档长度
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)
        
        # 初始化文档频率字典
        nd = self._initialize(corpus)
        # 计算逆文档频率（IDF）
        self._calc_idf(nd)

    def _initialize(self, corpus):
        """ 初始化文档词频词典
        """
        nd = {}         # 词语 -> 包含该词语的文档数
        num_doc = 0     # 文档总词数
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            # 计算每个文档中词语的频率
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            # 添加词语频率到列表中
            self.doc_freqs.append(frequencies)

            # 更新文档频率字典
            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1
            
            self.corpus_size += 1

        # 计算平均文档长度
        self.avgdl = num_doc / self.corpus_size
        return nd
    
    def _tokenize_corpus(self, corpus):
        """ 分词
        """
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        """ 计算逆文档频率（IDF）
        """
        raise NotImplementedError()

    def get_scores(self, query):
        """ 计算 query 与文档的相关性得分
        """
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        """ 计算 query 与一批文档的相关性得分
        """
        raise NotImplementedError()
    
    def get_top_n(self, query, documents, n=5):
        """ 获取 top-n
        """
        
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
 
        scores = self.get_scores(query)
        # 获取得分最高的n个文档的索引
        top_n = np.argsort(scores)[::-1][:n]
        # 根据索引获取文档
        return [documents[i] for i in top_n]
    
class BM25Okapi(BM25):
    """ 经典的BM25实现，通过设置IDF的下限来处理稀有词语的情况。
    """
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1    # 控制文档频率的影响程度
        self.b = b      # 控制文档长度的影响程度
        self.epsilon = epsilon  # IDF值的下限因子
        super().__init__(corpus, tokenizer)
        

    def _calc_idf(self, nd):
        """ 计算文档和语料库中术语的频率。
            该算法将 idf 值的下限设置为 eps *average_idf
        """
        idf_sum = 0         # 逆文档频率之和
        negative_idfs = []  # 存储IDF值小于0的词语
        for word, freq in nd.items():
            # 计算逆文档频率（IDF）
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        # 计算平均逆文档频率
        self.average_idf = idf_sum / len(self.idf)

        # 设置IDF值的下限
        eps = self.epsilon * self.average_idf
        # 将IDF值小于0的词语设置为下限值
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """ 计算 query 与文档的相关性得分。
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        
        return score
    
    def get_batch_scores(self, query, doc_ids):
        """ 计算查询与指定文档集的相关性得分。
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()

class BM25L(BM25):
    """ BM25的扩展版本, 通过引入ctd来调整文档长度的影响
    """
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        super().__init__(corpus, tokenizer)
        self.k1 = k1
        self.b = b
        self.delta = delta  # 调整IDF的计算

    def _calc_idf(self, nd):
        """ 计算文档和语料库中词语的逆文档频率（IDF）。
            log(N + 1) - log(freq + 0.5)； N是文档总数，freq是文档频率
        """
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        """ 计算 query 与文档的相关性得分。
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """ 计算查询与指定文档集的相关性得分。
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


class BM25Plus(BM25):
    """ BM25的扩展版本, 通过delta来进一步调整得分
    """
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        super().__init__(corpus, tokenizer)
        self.k1 = k1
        self.b = b
        self.delta = delta  # 调整得分计算

    def _calc_idf(self, nd):
        """ 计算文档和语料库中词语的逆文档频率（IDF）。
            log(N + 1) - log(freq)
        """
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        """ 计算 query 与文档的相关性得分。
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """ 计算查询与指定文档集的相关性得分。
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()
