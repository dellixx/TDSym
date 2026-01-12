import torch
from torch import nn
import numpy as np
from typing import Tuple
from .base import TKBCModel 
import torch.nn.functional as F

# ============================================================
# Math Utilities: Symplectic Preservation
# ============================================================
def expm_cayley(A: torch.Tensor) -> torch.Tensor:
    """
    Implements the Cayley Transform to map the Lie Algebra sp(2n) 
    to the Symplectic Lie Group Sp(2n).
    Formula: M = (I - A/2)^{-1} (I + A/2)
    """
    B, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
    half_A = 0.5 * A
    numerator = I + half_A
    denominator = I - half_A
    return torch.linalg.solve(denominator, numerator)

# ============================================================
# Module: Continuous Time Encoding
# ============================================================
class RotaryTimeGenerator(nn.Module):
    """
    Implements Rotary Time Embeddings (RoTE) for continuous time representation.
    """
    def __init__(self, d_model, base_freq=10000.0):
        super(RotaryTimeGenerator, self).__init__()
        self.d_model = d_model
        
        # Precompute frequencies for rotary encoding
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Base temporal vector initialization
        self.base_emb = nn.Parameter(torch.randn(1, d_model))
        nn.init.xavier_normal_(self.base_emb)

    def forward(self, t_vals):
        """
        Input:  Discrete time indices or timestamps.
        Output: Continuous rotary time embeddings.
        """
        batch_size = t_vals.shape[0]
        
        angles = t_vals.float().unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        
        x = self.base_emb.expand(batch_size, -1)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        
        # Apply rotation matrix logic
        rot_even = x_even * cos_vals - x_odd * sin_vals
        rot_odd  = x_even * sin_vals + x_odd * cos_vals
        
        rot = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return rot

# ============================================================
# Main Model: TDSym
# ============================================================
class TDSym(TKBCModel):
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, no_time_emb=False, init_size: float = 1e-2):
        super(TDSym, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.D = 2 * rank 
        

        # Entity embeddings represent generalized coordinates (q) and momenta (p)
        self.emb_e = nn.Embedding(sizes[0], self.D)
        
        # Relation embeddings serve as the basis for the Hamiltonian operator
        self.rel_embed = nn.Embedding(sizes[1], 2 * rank)

        # Discrete Time Embedding (Capture coarse-grained temporal patterns)
        self.time_embed = nn.Embedding(sizes[3], 2 * rank)
        
        # Continuous Time Encoding (Capture fine-grained continuity)
        self.time_encoder = RotaryTimeGenerator(d_model=2 * rank)
        
        # FiLM Generator (Feature-wise Linear Modulation)
        # Maps continuous time to modulation parameters (Gamma, Beta)
        self.film_generator = nn.Sequential(
            nn.Linear(2 * rank, 4 * rank),
        )
        
        # Initialization
        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        nn.init.xavier_uniform_(self.time_embed.weight)
        
        # Initialize FiLM to Identity (Gamma=0, Beta=0) for stable start
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        self.dt = 1.0

    def _get_modulated_matrix(self, r_idx, t_idx):
        """
        Constructs the time-dependent Hamiltonian matrix A(t).
        Logic: Combines Relation(r) and Time(t) via FiLM and Gating mechanisms.
        """
        # Retrieve static relation parameters
        r_vec = self.rel_embed(r_idx)
        
        # Retrieve temporal features
        t_discrete = self.time_embed(t_idx)        # Discrete Embedding
        t_RoTE = self.time_encoder(t_idx.float())  # Continuous RoTE
        
        # Generate FiLM affine parameters (Gamma, Beta)
        film_params = self.film_generator(t_RoTE)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        t_vec = (1.0 + gamma) * t_discrete + beta

        r_scale, r_bias = r_vec[:, :self.rank], r_vec[:, self.rank:]
        t_scale, t_bias = t_vec[:, :self.rank], t_vec[:, self.rank:]

        time_gate = torch.sigmoid(t_scale)
        eff_scale = r_scale * time_gate 
        eff_bias = r_bias + t_bias

        # Construct Hamiltonian Matrix Element A in sp(2n) (Section 3.2)
        # Structure: [ D_s,  D_b ]
        #            [-D_b, -D_s ]
        D_s = torch.diag_embed(eff_scale)
        D_b = torch.diag_embed(eff_bias)
        
        top = torch.cat([D_s, D_b], dim=2)
        bot = torch.cat([-D_b, -D_s], dim=2)
        A = torch.cat([top, bot], dim=1)
        
        return A, eff_scale, eff_bias

    def _get_evolution_operator(self, r_idx, t_idx):
        """
        Generates the Symplectic Evolution Operator M.
        """
        A_total, _, _ = self._get_modulated_matrix(r_idx, t_idx)
        # Apply Cayley Transform: A -> M \in Sp(2n)
        M = expm_cayley(A_total * self.dt)
        return M

    def get_queries(self, queries: torch.Tensor):
        """
        Inference function for evolving queries over time.
        """
        device = self.emb_e.weight.device
        queries = queries.to(device)
        
        h = self.emb_e(queries[:, 0]) 
        
        # Compute Evolution Operator M(t)
        M = self._get_evolution_operator(queries[:, 1], queries[:, 3])
        
        # Evolve State: h(t) = M(t) * h(t_0)
        q_evolved = torch.einsum("bij,bj->bi", M, h)
        
        return q_evolved

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """
        Twists (q, p) to (p, -q) for the symplectic inner product.
        """
        rhs = self.emb_e.weight.data[chunk_begin:chunk_begin + chunk_size]
        rhs_q, rhs_p = rhs[:, :self.rank], rhs[:, self.rank:]
        rhs_twisted = torch.cat([rhs_p, -rhs_q], dim=1)
        return rhs_twisted.transpose(0, 1)

    def forward(self, x):
        """
        Forward pass for training.
        """
        device = self.emb_e.weight.device; x = x.to(device)
        
        lhs = self.emb_e(x[:, 0])
        
        # Get Time-Aware Hamiltonian Matrix
        A_total, eff_scale, eff_bias = self._get_modulated_matrix(x[:, 1], x[:, 3])
        
        M = expm_cayley(A_total * self.dt)
        
        # Evolve Head Entity
        q_pred = torch.einsum("bij,bj->bi", M, lhs)
        
        # Score against Tail Entities (Symplectic Inner Product)
        # Score = Omega(h_evolved, h_tail)
        rhs_all = self.emb_e.weight
        rhs_q, rhs_p = rhs_all[:, :self.rank], rhs_all[:, self.rank:]
        rhs_twisted = torch.cat([rhs_p, -rhs_q], dim=1)
        logits = q_pred @ rhs_twisted.t()

        factors = (
            torch.norm(lhs, dim=1),
            torch.norm(eff_scale, dim=1),
            torch.norm(eff_bias, dim=1),
            torch.norm(self.emb_e(x[:, 2]), dim=1)
        )
        
        return logits, factors, None, None

    def score(self, x):
        """
        Computes the symplectic distance score between evolved head and tail.
        """
        q = self.get_queries(x)
        t = self.emb_e(x[:, 2])
        
        q_h, p_h = q[:, :self.rank], q[:, self.rank:]
        q_t, p_t = t[:, :self.rank], t[:, self.rank:]
        
        # Symplectic Inner Product: q_h * p_t - p_h * q_t
        return torch.sum(q_h * p_t - p_h * q_t, dim=1, keepdim=True)