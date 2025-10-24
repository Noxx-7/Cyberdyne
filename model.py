import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, n_heads, dropout=0.1):
        super().__init__()
        assert emb_size % n_heads == 0

        self.emb_size = emb_size
        self.n_heads = n_heads
        self.head_dim = emb_size // n_heads

        self.qkv = nn.Linear(emb_size, 3 * emb_size)
        self.out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, self.emb_size)
        out = self.out(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, emb_size, ff_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, ff_size)
        self.fc2 = nn.Linear(ff_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, n_heads, ff_size, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(emb_size, n_heads, dropout)
        self.ff = FeedForward(emb_size, ff_size, dropout)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x

class AdvancedLLM(nn.Module):
    def __init__(self, vocab_size, emb_size=768, n_layers=12, n_heads=12,
                 ff_size=3072, max_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, emb_size)
        self.pos_embed = nn.Embedding(max_len, emb_size)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_heads, ff_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, vocab_size, bias=False)

        self.token_embed.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        token_emb = self.token_embed(x)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)

        x = self.dropout(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.head(x)

        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save_model(self, path):
        config = {
            'vocab_size': self.vocab_size,
            'emb_size': self.emb_size,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'max_len': self.max_len,
            'state_dict': self.state_dict()
        }
        torch.save(config, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path, device='cpu'):
        config = torch.load(path, map_location=device)
        model = cls(
            vocab_size=config['vocab_size'],
            emb_size=config['emb_size'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_len=config['max_len']
        )
        model.load_state_dict(config['state_dict'])
        model.to(device)
        print(f"Model loaded from {path}")
        return model
