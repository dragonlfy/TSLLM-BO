# soft_dtw.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftDTW(nn.Module):
    """
    Differentiable Soft-DTW (batch) implementation.
    Reference: Cuturi & Blondel, 2017.
    Args
    ----
    gamma : float > 0
        Smoothing parameter; ->0 时趋近于经典 DTW，gamma 越大越平滑。
    normalize : bool
        若为 True，返回 normalized_soft_dtw = 0.5*(sdtw(x,x)+sdtw(y,y)-2*sdtw(x,y))
        这样可近似距离且对序列长度/幅值更稳。
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = False):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    @staticmethod
    def _pairwise_distances(x, y):
        # x,y: [B, L, C]  ->  cost: [B, L, L]
        B, L, C = x.shape
        x2 = (x ** 2).sum(-1).unsqueeze(-1)      # [B,L,1]
        y2 = (y ** 2).sum(-1).unsqueeze(-2)      # [B,1,L]
        xy = x @ y.transpose(-1, -2)             # [B,L,L]
        return x2 + y2 - 2 * xy                  # squared Euclid, >=0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim == 3, "expect [B,L,C]"
        B, L, _ = x.shape
        gamma = self.gamma

        D = self._pairwise_distances(x, y) / gamma  # scaled cost
        R = x.new_full((B, L + 2, L + 2), math.inf) # DP table
        R[:, 0, 0] = 0.

        for j in range(1, L + 1):
            Dj = D[:, j - 1]                        # [B,L]
            for i in range(1, L + 1):
                r0 = -R[:, i - 1, j - 1]            # three predecessors
                r1 = -R[:, i - 1, j    ]
                r2 = -R[:, i    , j - 1]
                rmax = torch.max(torch.stack((r0, r1, r2), dim=-1), dim=-1)[0]
                rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
                softmin = -(torch.log(rsum) + rmax)
                R[:, i, j] = Dj[:, i - 1] + softmin

        sdtw = R[:, L, L] * gamma                  # [B]

        if self.normalize:                         # 0.5(d(x,x)+d(y,y)-2d(x,y))
            diag = torch.diagonal(D, dim1=1, dim2=2)  # [B,L]
            sdtw_xx = (diag * 0).sum(dim=1)        # 0
            sdtw_yy = sdtw_xx.clone()
            return 0.5 * (sdtw_xx + sdtw_yy - 2 * sdtw)
        else:
            return sdtw
