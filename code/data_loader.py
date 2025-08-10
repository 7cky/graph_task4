import os
import numpy as np

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.entities = set()
        self.relations = set()
        self.train_data = self._load_triples('train.txt')
        self.valid_data = self._load_triples('valid.txt')
        self.test_data = self._load_triples('test.txt')
        self.entity2id = self._map_to_id(self.entities)
        self.rel2id = self._map_to_id(self.relations)

    def _load_triples(self, filename):
        triples = []
        with open(os.path.join(self.data_path, filename), 'r') as f:
        # with open(os.path.normpath(os.path.join(self.data_path, filename)), 'r') as f:
            for line in f:
                h, r, t = line.strip().split()
                self.entities.add(h)
                self.entities.add(t)
                self.relations.add(r)
                triples.append((h, r, t))
        return triples

    def _map_to_id(self, elements):
        return {elem: i for i, elem in enumerate(sorted(elements))}

    def get_id_triples(self, data_type='train'):
        """将三元组转换为id格式"""
        data = getattr(self, f'{data_type}_data')
        id_triples = []
        for h, r, t in data:
            id_triples.append((
                self.entity2id[h],
                self.rel2id[r],
                self.entity2id[t]
            ))
        return np.array(id_triples, dtype=np.int64)