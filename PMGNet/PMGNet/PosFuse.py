import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PositionEmbeddingRandom3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_dim = None
        self.register_buffer('positional_encoding_gaussian_matrix',
                             torch.empty(0),
                             persistent=False)

    def _init_embed(self, embed_dim: int):
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        scale = 1.0 / math.sqrt(3)
        mat = torch.randn(3, half_dim,
                          device=self.positional_encoding_gaussian_matrix.device) * scale
        self.register_buffer('positional_encoding_gaussian_matrix',
                             mat,
                             persistent=False)

    def forward(self,
                grid_size: Tuple[int, int, int],
                embed_dim: int) -> torch.Tensor:
        if self.embed_dim != embed_dim:
            self._init_embed(embed_dim)

        d, h, w = grid_size
        device = self.positional_encoding_gaussian_matrix.device
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, d, device=device),
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [N,3]
        proj = coords @ self.positional_encoding_gaussian_matrix  # [N, half_dim]
        proj = 2 * math.pi * proj
        pe = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)  # [N, embed_dim]
        C = pe.shape[1]
        pe = pe.view(d, h, w, C).permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,D,H,W]
        return pe


class PromptEncoder(nn.Module):


    def __init__(self, in_ch: int):
        super().__init__()
        embed_dim = in_ch       # 不翻倍通道，节省显存
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, embed_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(4, embed_dim // 2),
                         num_channels=embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1)
        )
        self.position_embed = PositionEmbeddingRandom3D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        feat = self.conv(x)
        pe = self.position_embed(feat.shape[2:], feat.shape[1])

        return feat + pe


class SpatialGate(nn.Module):


    def __init__(self, feat_dim: int, prompt_dim: int):
        super().__init__()

        self.fusion = nn.Conv3d(feat_dim + prompt_dim, 1, kernel_size=3, padding=1)

    def forward(self, E_B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:

        B2 = A.size(0)


        if E_B.shape[2:] != A.shape[2:]:
            E_B = F.interpolate(
                E_B,
                size=A.shape[2:],  # (D,H,W)
                mode="trilinear",
                align_corners=False,
            )


        B1 = E_B.size(0)
        if B1 == 1 and B2 > 1:

            E_B = E_B.expand(B2, -1, -1, -1, -1)


        fused = torch.cat([A, E_B], dim=1)
        attn = torch.sigmoid(self.fusion(fused))
        return attn


class ProbPromptFusion(nn.Module):


    def __init__(self):
        super().__init__()
        self.encoder = None
        self.gate = None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        if self.encoder is None or self.gate is None:
            feat_dim = A.shape[1]
            in_ch = B.shape[1]
            device = B.device
            self.encoder = PromptEncoder(in_ch=in_ch).to(device)
            self.gate = SpatialGate(feat_dim=feat_dim,
                                    prompt_dim=in_ch).to(device)


        E_B = self.encoder(B)
        attn = self.gate(E_B, A)
        return A * (1 + attn)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion = ProbPromptFusion().to(device)


    A = torch.randn(2, 8, 16, 16, 16, device=device)
    B = torch.randn(2, 1, 8, 8, 8, device=device)

    out = fusion(A, B)
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Output shape: {out.shape}  (应为 [2, 8, 16, 16, 16])")
