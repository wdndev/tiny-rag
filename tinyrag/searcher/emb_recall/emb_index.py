import faiss
import numpy as np

class EmbIndex:
    def __init__(self, index_dim: int) -> None:
        description = "HNSW64"
        measure = faiss.METRIC_L2
        # self.index = faiss.index_factory(index_dim, description, measure)
        self.index = faiss.IndexFlatL2(index_dim)
    def insert(self, emb: list):
        emb = np.array(emb, dtype=np.float32)  # 转换为 NumPy 数组
        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)  # 转换为 (1, d) 形状
        # print("Inserting emb: ", emb)
        # print("Inserting emb: ", emb.shape)
        self.index.add(emb)
        # print("Insertion successful")
    
    def batch_insert(self, embs: list):
        embs = np.array(embs, dtype=np.float32)  # 转换为 NumPy 数组
        if embs.ndim == 1:
            embs = np.expand_dims(embs, axis=0)  # 转换为 (1, d) 形状
        elif embs.ndim == 2 and embs.shape[0] == 1:
            embs = np.squeeze(embs, axis=0)  # 处理 (1, d) 形状
        print("Batch inserting embs: ", embs)
        self.index.add(embs)
        print("Batch insertion successful")

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def search(self, vec: list, num: int):
        vec = np.array(vec, dtype=np.float32)  # 转换为 NumPy 数组
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)  # 转换为 (1, d) 形状
        return self.index.search(vec, num)
