import os
import json

# from emb_index import EmbIndex
from tinyrag.searcher.emb_recall.emb_index import EmbIndex

class EmbRetriever:
    def __init__(self, index_dim: int, base_dir="data/db/faiss_idx") -> None:
        self.index_dim = index_dim
        self.invert_index = EmbIndex(index_dim)
        self.forward_index = []
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

    def insert(self, emb: list, doc: str):
        # print("Inserting document into index...")
        self.invert_index.insert(emb)
        self.forward_index.append(doc)
        # print("Document inserted")

    def save(self, index_name=""):
        self.index_name = index_name if index_name != "" else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)
        if not os.path.exists(self.index_folder_path):
            os.makedirs(self.index_folder_path, exist_ok=True)

        with open(self.index_folder_path + "/forward_index.txt", "w", encoding="utf8") as f:
            for data in self.forward_index:
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))

        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")
    
    def load(self, index_name=""):
        self.index_name = index_name if index_name != "" else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)

        self.invert_index = EmbIndex(self.index_dim)
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))

    def search(self, embs: list, top_n=5):
        search_res = self.invert_index.search(embs, top_n)
        recall_list = []
        for idx in range(top_n):
            recall_list.append((search_res[1][0][idx], self.forward_index[search_res[1][0][idx]], search_res[0][0][idx]))
        return recall_list
