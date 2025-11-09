from __future__ import annotations
from datetime import timedelta
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc as auc_metric,
    recall_score, precision_score, f1_score, confusion_matrix,)
from sklearn.metrics import fbeta_score

# Utility

def fast_hash(x: str, num_buckets: int = 50_000) -> int:
    """Deterministic hash to integer within [0, num_buckets)."""
    if x is None:
        return 0
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % num_buckets


def _next_pow2(n: int) -> int:
    """Return next power-of-two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def suggest_hash_buckets(n_unique: int, growth: float = 1.25, cap: Optional[int] = None) -> int:
    """Suggest a hash bucket size based on number of uniques."""
    base = int(max(1, np.ceil(n_unique * growth)))
    buckets = _next_pow2(base)
    if cap is not None:
        buckets = min(buckets, cap)
    return max(buckets, 2)


# Loss Functions

class FocalLoss(nn.Module):
    """Balanced focal loss."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()




# Sampler

def build_weighted_sampler(labels: pl.Series, power: float = 0.7) -> WeightedRandomSampler:
    """Weighted sampler for imbalance control."""
    y = labels.to_numpy()
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    weights = np.array([1.0 / (class_counts[c] ** power) for c in y], dtype=np.float32)
    weights = torch.as_tensor(weights)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# Preprocessing

#Done in the parquet files




# Dataset

@dataclass
class EncodedBatch:
    x_cat: torch.LongTensor
    x_cont: torch.FloatTensor
    y: torch.FloatTensor


def collate_encoded_batch(batch: List[EncodedBatch]) -> EncodedBatch:
    x_cat = torch.stack([b.x_cat for b in batch])
    x_cont = torch.stack([b.x_cont for b in batch]) if batch[0].x_cont.numel() > 0 else torch.empty((len(batch), 0))
    y = torch.stack([b.y for b in batch])
    return EncodedBatch(x_cat, x_cont, y)


class TabAMLDataset(Dataset):
    """Encodes categorical + continuous features; adaptive hash for Sender/Receiver accounts."""
    def __init__(self, df, cat_cols, cont_cols, label_col,
                 encoders=None, scaler=None, fit=True,
                 hash_bucket_sizes: Optional[Dict[str, int]] = None,
                 hash_growth: float = 1.25,
                 hash_cap: Optional[Dict[str, int]] = None):
        super().__init__()
        self.df = df
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_col = label_col
        self.y = df[label_col].cast(pl.Float32).to_numpy()
        self.encoders = {} if fit or encoders is None else encoders

        high_card_cols = {"Sender_account", "Receiver_account"}
        if fit:
            self.hash_bucket_sizes = {}
            for col in (high_card_cols & set(cat_cols)):
                n_unique = int(df[col].n_unique())
                cap_for_col = (hash_cap or {}).get(col) if hash_cap else None
                self.hash_bucket_sizes[col] = suggest_hash_buckets(n_unique, growth=hash_growth, cap=cap_for_col)
        else:
            if hash_bucket_sizes is None:
                raise ValueError("hash_bucket_sizes must be provided when fit=False.")
            self.hash_bucket_sizes = dict(hash_bucket_sizes)

        encoded_cols = []
        for col in cat_cols:
            vals = df[col].cast(pl.Utf8).to_list()
            if col in high_card_cols:
                buckets = self.hash_bucket_sizes[col]
                arr = np.fromiter((fast_hash(v, num_buckets=buckets) for v in vals), dtype=np.int64)
                arr %= buckets
                encoded_cols.append(arr)
                continue

            if fit or col not in self.encoders:
                le = LabelEncoder()
                le.fit(list(vals) + ["UNK"])
                self.encoders[col] = le
            le = self.encoders[col]
            mapped = [v if v in le.classes_ else "UNK" for v in vals]
            encoded_cols.append(le.transform(mapped).astype(np.int64))

        self.x_cat = np.stack(encoded_cols, axis=1).astype(np.int64)

        if cont_cols:
            arr = df.select(cont_cols).to_numpy()
            if fit or scaler is None:
                self.scaler = StandardScaler()
                self.x_cont = self.scaler.fit_transform(arr).astype(np.float32)
            else:
                self.scaler = scaler
                self.x_cont = self.scaler.transform(arr).astype(np.float32)
        else:
            self.scaler = None
            self.x_cont = np.empty((len(df), 0), dtype=np.float32)

        self.category_sizes = [
            self.hash_bucket_sizes[c] if c in high_card_cols else len(self.encoders[c].classes_)
            for c in cat_cols
        ]

    def __len__(self): return len(self.y)
    def __getitem__(self, idx: int):
        cat = torch.from_numpy(self.x_cat[idx]).long()
        cont = torch.from_numpy(self.x_cont[idx]).float() if self.x_cont.size > 0 else torch.empty(0)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return EncodedBatch(cat, cont, y)


# Model

class ResidualAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or d_model * 4
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, d_model), nn.Dropout(dropout)
        )
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.scale, self.drop = (d_model // nhead) ** -0.5, nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, prev_attn=None):
        b, t, d = x.shape
        h = self.nhead
        qkv = self.qkv(self.norm1(x))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k, v = [z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v)]
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if prev_attn is not None:
            attn += prev_attn
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.attn_out(self.attn_dropout(out))
        x = x + self.ffn(self.norm2(x))
        return x, attn.detach()


class ResidualAttentionEncoder(nn.Module):
    def __init__(self, n_layers, d_model, nhead, dropout):
        super().__init__()
        self.layers = nn.ModuleList([ResidualAttentionLayer(d_model, nhead, dropout=dropout) for _ in range(n_layers)])
    def forward(self, x):
        attn = None
        for layer in self.layers:
            x, attn = layer(x, attn)
        return x


class TokenDropout(nn.Module):
    """Feature-level dropout."""
    def __init__(self, p: float = 0.40):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p <= 0: return x
        mask = (torch.rand(x.size(0), x.size(1), device=x.device) > self.p).float()
        return x * mask.unsqueeze(-1)


class TabAMLModel(nn.Module):
    def __init__(self, category_sizes, cont_dim, embedding_dim=48, shared_ratio=1/6,
                 num_heads=4, num_layers1=2, num_layers2=2, dropout=0.25, micro_indices=(0,1)):
        super().__init__()
        self.num_cat = len(category_sizes)
        self.cont_dim = cont_dim
        self.embedding_dim = embedding_dim
        self.shared_dim = max(1, int(embedding_dim * shared_ratio))
        self.indiv_dim = embedding_dim - self.shared_dim
        self.micro_indices = micro_indices

        self.shared_emb = nn.Parameter(torch.randn(self.shared_dim))
        self.embs = nn.ModuleList([nn.Embedding(sz, self.indiv_dim) for sz in category_sizes])
        self.micro = ResidualAttentionEncoder(num_layers1, embedding_dim, num_heads, dropout)
        self.macro = ResidualAttentionEncoder(num_layers2, embedding_dim, num_heads, dropout)
        self.token_dropout = TokenDropout(p=0.40)

        self.cont_proj = nn.Identity() if cont_dim == 0 else nn.Sequential(
            nn.Linear(cont_dim, cont_dim), nn.LayerNorm(cont_dim))

        flat_dim = self.num_cat * embedding_dim + cont_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, flat_dim // 4), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(flat_dim // 4, 1)
        )

    def embed_categorical(self, x_cat):
        b = x_cat.size(0)
        shared = self.shared_emb.expand(b, self.shared_dim)
        tokens = [torch.cat([shared, emb(x_cat[:, i])], dim=1) for i, emb in enumerate(self.embs)]
        return torch.stack(tokens, dim=1)

    def forward(self, x_cat, x_cont):
        cat_tokens = self.embed_categorical(x_cat)
        cat_tokens = self.token_dropout(cat_tokens)
        micro_tokens = cat_tokens[:, list(self.micro_indices), :]
        micro_out = self.micro(micro_tokens)
        for j, idx in enumerate(self.micro_indices):
            cat_tokens[:, idx, :] = micro_out[:, j, :]
        macro_out = self.macro(cat_tokens)
        flat = macro_out.reshape(macro_out.size(0), -1)
        cont = self.cont_proj(x_cont)
        logits = self.mlp(torch.cat([flat, cont], dim=1)).squeeze(1)
        return logits


# Training / Evaluation

def train_epoch(model, loader, optimizer, loss_fn, device, max_norm=2.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x_cat, x_cont, y = batch.x_cat.to(device), batch.x_cont.to(device), batch.y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cat, x_cont)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    total, n = 0.0, 0
    y_all, p_all = [], []
    for batch in loader:
        x_cat, x_cont, y = batch.x_cat.to(device), batch.x_cont.to(device), batch.y.to(device)
        logits = model(x_cat, x_cont)
        loss = loss_fn(logits, y)
        probs = torch.sigmoid(logits).cpu().numpy()
        total += loss.item() * y.size(0)
        n += y.size(0)
        y_all.extend(y.cpu().numpy())
        p_all.extend(probs)
    y_all, p_all = np.array(y_all), np.array(p_all)
    avg_loss = total / max(n, 1)
    auc_roc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) > 1 else np.nan
    prec, rec, _ = precision_recall_curve(y_all, p_all)
    auc_pr = auc_metric(rec, prec)
    preds = (p_all >= threshold).astype(int)
    rec_v, prec_v, f1_v = recall_score(y_all, preds), precision_score(y_all, preds), f1_score(y_all, preds)
    tn, fp, fn, tp = confusion_matrix(y_all, preds).ravel().tolist()
    return avg_loss, auc_roc, auc_pr, rec_v, prec_v, f1_v, (tn, fp, fn, tp)

def find_optimal_threshold(probs, labels, beta=2,
                           threshold_search_range=np.arange(0.05, 0.95, 0.05)):
    best_f2 = -1
    best_threshold = 0.5

    for threshold in threshold_search_range:
        preds = (probs >= threshold).astype(int)
        f2 = fbeta_score(labels, preds, beta=beta, average="binary", zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    return best_threshold, best_f2


