import torch
import torch.nn as nn
from typing import Tuple
from .base import TKBCModel

class TeLM(TKBCModel):

    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1
    ):
        super(TeLM, self).__init__()
        self.sizes = sizes
        self.rank = rank
        
        # TeLM embeddings have 4 components (scalar, e1, e2, e1e2), so dim = 4 * rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]] # entities, relations, time
        ])
        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        


        self.no_time_emb = no_time_emb
        self.time_granularity = time_granularity

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)
        
        lhs = lhs.chunk(4, dim=1)
        rel = rel.chunk(4, dim=1)
        rhs = rhs.chunk(4, dim=1)
        time = time.chunk(4, dim=1)

        A = rel[0]*time[0] + rel[1]*time[1] + rel[2]*time[2] - rel[3]*time[3]
        B = rel[0]*time[1] + rel[1]*time[0] - rel[2]*time[3] + rel[3]*time[2]
        C = rel[0]*time[2] + rel[2]*time[0] + rel[1]*time[3] - rel[3]*time[1]
        D = rel[1]*time[2] - rel[2]*time[1] + rel[0]*time[3] + rel[3]*time[0]
        
        full_rel = (A, B, C, D)

        W = lhs[0]*full_rel[0] + lhs[1]*full_rel[1] + lhs[2]*full_rel[2] - lhs[3]*full_rel[3]
        X = lhs[0]*full_rel[1] + lhs[1]*full_rel[0] - lhs[2]*full_rel[3] + lhs[3]*full_rel[2]
        Y = lhs[0]*full_rel[2] + lhs[2]*full_rel[0] + lhs[1]*full_rel[3] - lhs[3]*full_rel[1]
        Z = lhs[1]*full_rel[2] - lhs[2]*full_rel[1] + lhs[0]*full_rel[3] + lhs[3]*full_rel[0]

        return torch.sum(W*rhs[0] - X*rhs[1] - Y*rhs[2] + Z*rhs[3], 1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)

        lhs = lhs.chunk(4, dim=1)
        rel = rel.chunk(4, dim=1)
        rhs = rhs.chunk(4, dim=1)
        time = time.chunk(4, dim=1)

        A = rel[0]*time[0] + rel[1]*time[1] + rel[2]*time[2] - rel[3]*time[3]
        B = rel[0]*time[1] + rel[1]*time[0] - rel[2]*time[3] + rel[3]*time[2]
        C = rel[0]*time[2] + rel[2]*time[0] + rel[1]*time[3] - rel[3]*time[1]
        D = rel[1]*time[2] - rel[2]*time[1] + rel[0]*time[3] + rel[3]*time[0]
        full_rel = (A, B, C, D)
        
        W = lhs[0]*full_rel[0] + lhs[1]*full_rel[1] + lhs[2]*full_rel[2] - lhs[3]*full_rel[3]
        X = lhs[0]*full_rel[1] + lhs[1]*full_rel[0] - lhs[2]*full_rel[3] + lhs[3]*full_rel[2]
        Y = lhs[0]*full_rel[2] + lhs[2]*full_rel[0] + lhs[1]*full_rel[3] - lhs[3]*full_rel[1]
        Z = lhs[1]*full_rel[2] - lhs[2]*full_rel[1] + lhs[0]*full_rel[3] + lhs[3]*full_rel[0]

        to_score = self.embeddings[0].weight
        to_score = to_score.chunk(4, dim=1)

        scores = (
            W @ to_score[0].t() -
            X @ to_score[1].t() -
            Y @ to_score[2].t() +
            Z @ to_score[3].t()
        )

        regularizers = (
            torch.sqrt(lhs[0]**2 + lhs[1]**2 + lhs[2]**2 + lhs[3]**2),
            torch.sqrt(full_rel[0]**2 + full_rel[1]**2 + full_rel[2]**2 + full_rel[3]**2),
            torch.sqrt(rhs[0]**2 + rhs[1]**2 + rhs[2]**2 + rhs[3]**2)
        )

        
        time_weight = self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        
        dummy_phase = torch.zeros_like(time_weight)
        
        dummy_flow = torch.tensor(0.0, device=scores.device)

        return scores, regularizers, time_weight, dummy_phase

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity)
        
        lhs = lhs.chunk(4, dim=1)
        rel = rel.chunk(4, dim=1)
        time = time.chunk(4, dim=1)

        A = lhs[0]*rel[0] + lhs[1]*rel[1] + lhs[2]*rel[2] - lhs[3]*rel[3]
        B = lhs[0]*rel[1] + lhs[1]*rel[0] - lhs[2]*rel[3] + lhs[3]*rel[2]
        C = lhs[0]*rel[2] + lhs[2]*rel[0] + lhs[1]*rel[3] - lhs[3]*rel[1]
        D = lhs[1]*rel[2] - lhs[2]*rel[1] + lhs[0]*rel[3] + lhs[3]*rel[0]
        
        W = A*time[0] + B*time[1] + C*time[2] - D*time[3]
        X = -A*time[1] - B*time[0] + C*time[3] - D*time[2]
        Y = -A*time[2] - C*time[0] - B*time[3] + D*time[1]
        Z = B*time[2] - C*time[1] + A*time[3] + D*time[0]

        return torch.cat([W, X, Y, Z], 1)