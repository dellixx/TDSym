import torch
import torch.nn as nn
from typing import Tuple
from .base import TKBCModel

class TCompoundE(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TCompoundE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

    @staticmethod
    def has_time():
        return True

    def score(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        # 逻辑: rt[0] 是 scaling 因子, rt[1] 是 translation 因子
        rt = (rel[0] + time[0]) * time[1], rel[1]
        
        return torch.sum(
            ((lhs[0] + rt[1]) * rt[0]) * rhs[0], 
            1, keepdim=True
        )

    def forward(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = (rel[0] + time[0]) * time[1], rel[1]

        scores = ((lhs[0] + rt[1]) * rt[0]) @ right[0].t()

        regularizers = (
            torch.sqrt(lhs[0] ** 2),
            torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
            torch.sqrt(rhs[0] ** 2)
        )

    
        time_weight = self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        
        dummy_phase = torch.zeros_like(time_weight)
        
        dummy_flow = torch.tensor(0.0, device=scores.device)

        return scores, regularizers, time_weight, dummy_phase

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        rt = (rel[0] + time[0]) * time[1], rel[1]
        
        return (lhs[0] + rt[1]) * rt[0]