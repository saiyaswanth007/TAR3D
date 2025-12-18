import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_func
from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor

# Re-use RMSNorm from gpt.py or define it here if needed. 
# Defining it here to be self-contained for the modules.
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super(SwiGLU, self).__init__()
        if hidden_dim is None:
            hidden_dim = int(8 / 3 * embed_dim)
            # Round to multiple of 256
            hidden_dim = 256 * ((hidden_dim + 255) // 256)
            
        self.proj1 = nn.Linear(embed_dim, hidden_dim, bias=False)  # XWG
        self.proj2 = nn.Linear(embed_dim, hidden_dim, bias=False)  # XW1
        self.proj3 = nn.Linear(hidden_dim, embed_dim, bias=False)  # W2

    def forward(self, x):
        x_proj1 = F.silu(self.proj1(x))
        x_proj2 = self.proj2(x)
        x_glu = x_proj1 * x_proj2
        output = self.proj3(x_glu)
        return output

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=10000):
        super().__init__()
        # half-truncate RoPE
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x_BTHD, offset=0):
        seq_len = x_BTHD.size(-3)
        # Ensure we don't go out of bounds
        if offset + seq_len > self.max_seq_len:
             # Fallback or error, but for now let's just clamp or assume it fits
             # In a real scenario, you might want to extend the buffer dynamically
             pass

        cos = self.cos[None, offset:offset + seq_len, None, :]
        sin = self.sin[None, offset:offset + seq_len, None, :]

        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x_BTHD)

class MultiheadFlashrope(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, causal=True):
        bsz, seq_len, embed_dim = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        q = self.rotary(q)
        k = self.rotary(k)

        # Flash Attention
        # Note: flash_attn_func expects (batch, seq_len, nheads, headdim)
        attn_output = flash_attn_func(q, k, v, causal=causal)

        attn_output = attn_output.contiguous().view(bsz, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

class MultiheadFlashlinearrope(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.rotary = Rotary(self.head_dim) # Use external RoPE
        self.rms_norm = RMSNorm(embed_dim, eps=1e-5)
        
        slope_rate = _build_slope_tensor(self.num_heads)
        self.register_buffer("slope_rate", slope_rate, persistent=False)

    def forward(self, x, freqs_cis=None, causal=True):
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        
        # Transpose for Lightning Attention: (B, H, N, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Lightning Attention
        # Note: This naturally handles zero-padding as described in the plan.
        attn_output = lightning_attn_func(q, k, v, self.slope_rate)

        # Reshape back
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        attn_output = self.rms_norm(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output

class DifferentialTransformerBlockrope(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        self.attn = MultiheadFlashrope(embed_dim, num_heads)
        self.causal = causal
        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x):
        attn_out = self.attn(self.norm1(x), causal=self.causal)
        x = x + attn_out
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        return x

class DifferentialTransformerBlocklinearrope(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        self.attn = MultiheadFlashlinearrope(embed_dim, num_heads)
        self.causal = causal
        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x):
        attn_out = self.attn(self.norm1(x), causal=self.causal)
        x = x + attn_out
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        return x
