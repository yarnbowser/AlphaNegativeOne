import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
    


class ConvNet2D(nn.Module):
    def __init__(self, piece_dim, board_size):
        super().__init__()

        convolved_size = piece_dim * 4 * board_size

        self.model = nn.Sequential(
            nn.Conv2d(piece_dim, piece_dim * 2, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.Conv2d(piece_dim * 2, piece_dim * 4, kernel_size=3, padding="same"),
            nn.GELU(),

            nn.Flatten(),

            nn.Linear(convolved_size, convolved_size // 2),
            nn.GELU(),
            nn.Linear(convolved_size // 2, 1),
            nn.Sigmoid() # for probabilities
        )

    def forward(self, board: torch.Tensor):
        return self.model(board)
    



class LearnablePositionalEncoding(nn.Module):
    def __init__(self, piece_dim: int, board_size: int=64):
        super().__init__()

        self.white_encodings = nn.Parameter(torch.zeros(1, board_size, piece_dim))
        self.black_encodings = nn.Parameter(torch.zeros(board_size, piece_dim))

        nn.init.normal_(self.white_encodings, mean=0.0, std=0.02)
        nn.init.normal_(self.black_encodings, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, is_white: bool) -> torch.Tensor:    
        return x + (self.white_encodings if is_white else self.black_encodings)
    


class QKVAttention(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        n_heads: int = 1,
        proj_dim: int | None = None,
    ):
        super().__init__()

        if proj_dim is None:
            proj_dim = piece_dim

        assert proj_dim % n_heads == 0

        self.query = nn.Linear(piece_dim, proj_dim)
        self.key = nn.Linear(piece_dim, proj_dim)
        self.value = nn.Linear(piece_dim, proj_dim)

        # output projection
        self.proj = nn.Linear(proj_dim, piece_dim)

        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(proj_dim // n_heads)

    def forward(self, x: torch.Tensor):
        n = self.n_heads

        q = rearrange(self.query(x), '... s (n h) -> ... n s h', n=n) # Shape: (..., n, s, h)
        k = rearrange(self.key(x), '... s (n h) -> ... n s h', n=n)
        v = rearrange(self.value(x), '... s (n h) -> ... n s h', n=n)

        att = F.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1) # Shape: (..., n, s, s)

        y = rearrange(att @ v, '... n s h -> ... s (n h)') # Shape: (..., s, p)

        # output projection
        return self.proj(y), att



class TransformerBlock(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        n_heads: int = 1,
        proj_dim: int | None = None,
        ffn_expansion: int = 4,
    ):
        super().__init__()

        self.attention = QKVAttention(
            piece_dim=piece_dim,
            n_heads=n_heads,
            proj_dim=proj_dim,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(piece_dim, piece_dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(piece_dim * ffn_expansion, piece_dim)
        )

        self.layer_norm_1 = nn.LayerNorm(piece_dim)
        self.layer_norm_2 = nn.LayerNorm(piece_dim)

    def forward(self, x: torch.Tensor):
        y, att = self.attention(x)
        x = self.layer_norm_1(x + y)

        ffn_out = self.feed_forward(x)
        return self.layer_norm_2(x + ffn_out)


class Transformer(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        board_size: int,
        n_blocks: int = 4,
        n_heads: int = 1,
        proj_dim: int | None = None,
        ffn_expansion: int = 4,
    ):
        super().__init__()

        self.piece_dim = piece_dim
        self.board_size = board_size

        self.pe = LearnablePositionalEncoding(piece_dim, board_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                piece_dim=piece_dim,
                n_heads=n_heads,
                proj_dim=proj_dim,
                ffn_expansion=ffn_expansion,
            )
            for _ in range(n_blocks)
        ])

        self.probability_head = ConvNet2D(piece_dim, board_size)


    def forward(self, board: torch.Tensor, is_white: bool):
        if board.dim() == 3:
            board = board.unsqueeze(0)

        b, h, w, p = board.size()

        x = rearrange(board, 'b h w p -> b (h w) p') # Shape: (b, h*w, piece_dim)
        x = self.pe(x, is_white)

        for block in self.blocks:
            x = block(x)
                
        x = rearrange(x, 'b (h w) p -> b p h w', h=h) # Shape: (b, h*w, piece_dim)

        return self.probability_head(x)







