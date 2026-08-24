
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP3(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, outputdim =1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  
            nn.Linear(hidden_dim, outputdim)   # predicts clean y (scalar)
        )

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.net(x)  # [B,1] predicted y

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ff(x))

class MLP1(nn.Module):
    """
    Keeps your original input/output structure:
      - __init__(input_dim, hidden_dim=256, outputdim=2)
      - forward(*inputs): concatenates along dim=1
      - returns [B, outputdim]

    But makes the *internals* similar to the tutorial:
      - separate projections for "x part" and "time part"
      - sinusoidal embedding for the first scalar input (time)
      - several residual-free blocks at width hidden_dim
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        outputdim: int = 2,
        layers: int = 5,
        t_emb_dim: int = 256,
        max_positions: int = 10000,
    ):
        super().__init__()
        assert input_dim >= 2, "Expected at least a scalar time + x dims"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.outputdim = outputdim
        self.layers = layers
        self.t_emb_dim = t_emb_dim
        self.max_positions = max_positions

        # We assume your concatenated input looks like: [time_scalar, rest_of_features]
        self.x_dim = input_dim - 1

        self.in_projection = nn.Linear(self.x_dim, hidden_dim)
        self.t_projection = nn.Linear(t_emb_dim, hidden_dim)
        self.blocks = nn.Sequential(*[Block(hidden_dim) for _ in range(layers)])
        self.out_projection = nn.Linear(hidden_dim, outputdim)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal embedding like the tutorial.
        t: [B] (typically in [0,1])
        returns: [B, t_emb_dim]
        """
        t = t * self.max_positions
        half = self.t_emb_dim // 2
        # handle tiny t_emb_dim safely
        if half <= 1:
            return t[:, None].repeat(1, self.t_emb_dim)

        scale = math.log(self.max_positions) / (half - 1)
        freqs = torch.arange(half, device=t.device, dtype=t.dtype).mul(-scale).exp()
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.t_emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)  # [B, input_dim]
        t = x[:, 0]                   # [B]
        feats = x[:, 1:]              # [B, input_dim-1]

        h = self.in_projection(feats)
        h = h + self.t_projection(self.time_embedding(t))
        h = self.blocks(h)
        return self.out_projection(h)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, outputdim=1, s_emb_dim=32, max_positions=10000):
        """
        Expects input layout: [s, u, x...]
          - s (scalar) gets sinusoidal embedding
          - u (scalar) is kept raw (no embedding)
          - x... are the remaining features (e.g., Xt)
        """
        super().__init__()
        assert input_dim >= 2, "Need at least [s, u] + features"
        self.s_emb_dim = s_emb_dim
        self.max_positions = max_positions

        x_dim = input_dim - 2  # excluding s and u

        # After embedding, the effective input becomes: [s_emb (s_emb_dim), u (1), x (x_dim)]
        eff_in = s_emb_dim + 1 + x_dim

        self.net = nn.Sequential(
            nn.Linear(eff_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, outputdim),
        )

    def s_embedding(self, s: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal embedding for s in [0,1).
        s: [B] or [B,1]
        returns: [B, s_emb_dim]
        """
        if s.dim() == 2:
            s = s[:, 0]
        s = s * self.max_positions

        half = self.s_emb_dim // 2
        if half <= 0:
            return s[:, None]

        scale = math.log(self.max_positions) / max(half - 1, 1)
        freqs = torch.arange(half, device=s.device, dtype=s.dtype).mul(-scale).exp()
        args = s[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.s_emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim] with layout [s, u, x...]
        """
        s = x[:, 0:1]     # [B,1]
        u = x[:, 1:2]     # [B,1]  (raw scalar)
        feats = x[:, 2:]  # [B, x_dim]

        s_emb = self.s_embedding(s)  # [B, s_emb_dim]
        x_eff = torch.cat([s_emb, u, feats], dim=1)
        return self.net(x_eff)
