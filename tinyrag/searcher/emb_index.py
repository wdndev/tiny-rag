import faiss
from loguru import logger

class EmbIndex:
    def __init__(self) -> None:
        self.index = ""

    def build(self, index_dim: int):
        description = "HNSW64"
        measure = faiss.METRIC_L2
        self.index = faiss.index_factory(index_dim, description, measure)

    def insert(self, emb):
        self.index.add(emb)
    
    def batch_insert(self, embs):
        self.index.add(embs)

    def load(self, path):
        self.index = faiss.read_index(path)

    def save(self, path):
        faiss.write_index(self.index, path)

    def search(self, vec, num):
        # id, distance
        return self.index.search(vec, num)
