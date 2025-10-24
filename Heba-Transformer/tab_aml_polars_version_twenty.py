
from __future__ import annotations
from datetime import timedelta
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc as auc_metric,
    recall_score, precision_score, f1_score, confusion_matrix,
)


def fast_hash(x: str, num_buckets: int = 50_000) -> int:
    if x is None:
        return 0
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % num_buckets

class FocalLoss(nn.Module):
    def __init__(self, alpha=4.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()



def preprocess_saml_d(df: pl.DataFrame) -> pl.DataFrame:
    
    

    #Ensure Date and Time are strings for concatenation ---
    # Handle case where Date is already a date/datetime
    if df["Date"].dtype != pl.Utf8:
        df = df.with_columns(
            pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date")
        )
    #Convert Time to string if needed
    df = df.with_columns(pl.col("Time").cast(pl.Utf8, strict=False))

    #Combine Date + Time into a single timestamp (robust parsing) ---
    dt_str = pl.concat_str(
        [pl.col("Date").str.strip_chars(), pl.col("Time").str.strip_chars()],
        separator=" "
    )

    #Handle both "%Y-%m-%d %H:%M:%S" and "%Y-%m-%d %H:%M" formats
    ts_hms = dt_str.str.strptime(pl.Datetime(), "%Y-%m-%d %H:%M:%S", strict=False)
    ts_hm  = dt_str.str.strptime(pl.Datetime(), "%Y-%m-%d %H:%M", strict=False)

    #Combine both parsing attempts
    df = df.with_columns(
        pl.coalesce([ts_hms, ts_hm]).alias("timestamp")
    ).drop_nulls(["timestamp"])

    #Temporal features ---
    df = df.with_columns([
        pl.col("timestamp").dt.day().alias("day"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        (pl.col("timestamp").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
    ])

    #Log-transform Amount safely ---
    df = df.with_columns(
        pl.when(pl.col("Amount").is_not_null())
          .then(pl.col("Amount").cast(pl.Float64).log1p())
          .otherwise(None)
          .alias("amount_log")
    )

    #Regenerate pure Date for splitting ---
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("Date")
    )
    

    #Drop redundant columns ---
    df = df.drop(["Laundering_type", "Time", "Amount"], strict=False)

    return df


def recast(df: pl.DataFrame) -> pl.DataFrame:
    
    exclude = ["Sender_account", "Receiver_account"]
    for col in df.columns:
        if col not in exclude:
            dtype = df[col].dtype
            if dtype in (pl.Int64, pl.Int32):
                maxval = df[col].drop_nulls().max()
                if maxval is not None:
                    if maxval < 127:
                        df = df.with_columns(pl.col(col).cast(pl.Int8))
                    elif maxval < 32767:
                        df = df.with_columns(pl.col(col).cast(pl.Int16))
                    elif maxval < 2147483647:
                        df = df.with_columns(pl.col(col).cast(pl.Int32))
    return df


def custom_split_polars(df: pl.DataFrame, validation_days: int = 70, test_days: int = 35):
    """
    Chronological split (train → val → test) based on calendar days from earliest to latest.
    Ensures non-overlapping sequential windows.
    """
    # Convert to datetime if needed
    dtype = df["Date"].dtype
    if dtype == pl.Utf8:
        df = df.with_columns(pl.col("Date").str.strptime(pl.Datetime(), "%Y-%m-%d", strict=False))
    elif dtype == pl.Date:
        df = df.with_columns(pl.col("Date").cast(pl.Datetime()))

    df = df.sort("Date")

    min_date = df.select(pl.col("Date").min()).item()
    max_date = df.select(pl.col("Date").max()).item()
    total_days = (max_date - min_date).days

    # Split: oldest → newest
    test_cutoff = max_date - timedelta(days=test_days)
    val_cutoff = test_cutoff - timedelta(days=validation_days)

    train_df = df.filter(pl.col("Date") < pl.lit(val_cutoff))
    val_df   = df.filter((pl.col("Date") >= pl.lit(val_cutoff)) & (pl.col("Date") < pl.lit(test_cutoff)))
    test_df  = df.filter(pl.col("Date") >= pl.lit(test_cutoff))

    print(f"Split complete:")
    print(f"  Train: {train_df.height} rows (until {val_cutoff.date()})")
    print(f"  Val:   {val_df.height} rows ({val_cutoff.date()}–{test_cutoff.date()})")
    print(f"  Test:  {test_df.height} rows (after {test_cutoff.date()})")

    return train_df, val_df, test_df


@dataclass
class EncodedBatch:
    x_cat: torch.LongTensor
    x_cont: torch.FloatTensor
    y: torch.FloatTensor


def collate_encoded_batch(batch: List[EncodedBatch]) -> EncodedBatch:
    x_cat = torch.stack([b.x_cat for b in batch])
    x_cont = torch.stack([b.x_cont for b in batch]) if batch[0].x_cont.numel() > 0 else torch.empty((len(batch), 0))
    y = torch.stack([b.y for b in batch])
    return EncodedBatch(x_cat=x_cat, x_cont=x_cont, y=y)


class TabAMLDataset(Dataset):
    """Encodes categorical and continuous features with fitted encoders/scaler."""
    def __init__(self, df, cat_cols, cont_cols, label_col,
                 encoders=None, scaler=None, fit=True):
        super().__init__()
        self.df = df
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_col = label_col
        self.y = self.df[label_col].cast(pl.Float32).to_numpy()
        self.encoders = {} if fit or encoders is None else encoders

        #encode categoricals ---
        encoded_cols = []
        for col in self.cat_cols:
            col_vals = self.df[col].cast(pl.Utf8).to_list()
            if col in ("Sender_account", "Receiver_account"):
                hashed = np.fromiter((fast_hash(v) for v in col_vals), dtype=np.int64, count=len(col_vals))
                encoded_cols.append(hashed)
                continue
            if fit or col not in self.encoders:
                le = LabelEncoder()
                le.fit(list(col_vals) + ["UNK"])
                self.encoders[col] = le
            le = self.encoders[col]
            mapped = [v if v in le.classes_ else "UNK" for v in col_vals]
            encoded_cols.append(le.transform(mapped).astype(np.int64))
        self.x_cat = np.stack(encoded_cols, axis=1).astype(np.int64)

        # Scale continuous ---
        if self.cont_cols:
            cont_arr = self.df.select(self.cont_cols).to_numpy()
            if fit or scaler is None:
                self.scaler = StandardScaler()
                self.x_cont = self.scaler.fit_transform(cont_arr).astype(np.float32)
            else:
                self.scaler = scaler
                self.x_cont = self.scaler.transform(cont_arr).astype(np.float32)
        else:
            self.scaler = None
            self.x_cont = np.empty((len(self.df), 0), dtype=np.float32)

        self.category_sizes = [
            50_000 if c in ("Sender_account", "Receiver_account") else len(self.encoders[c].classes_)
            for c in self.cat_cols
        ]

    def __len__(self): return len(self.y)
    def __getitem__(self, idx: int) -> EncodedBatch:
        cat = torch.from_numpy(self.x_cat[idx]).long()
        cont = torch.from_numpy(self.x_cont[idx]).float() if self.x_cont.size > 0 else torch.empty(0)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return EncodedBatch(x_cat=cat, x_cont=cont, y=y)



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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.scale = (d_model // nhead) ** -0.5
        self.drop = nn.Dropout(dropout)

    def forward(self, x, prev_attn=None):
        b, t, d = x.shape
        h = self.nhead
        qkv = self.qkv(self.norm1(x))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if prev_attn is not None: attn += prev_attn
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.attn_out(self.drop(out))
        x = x + self.ffn(self.norm2(x))
        return x, attn.detach()


class ResidualAttentionEncoder(nn.Module):
    def __init__(self, n_layers, d_model, nhead, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualAttentionLayer(d_model, nhead, dropout=dropout) for _ in range(n_layers)]
        )
    def forward(self, x):
        attn = None
        for layer in self.layers:
            x, attn = layer(x, attn)
        return x


class TabAMLModel(nn.Module):
    def __init__(self, category_sizes, cont_dim, embedding_dim=64, shared_ratio=1/6,
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
        self.cont_proj = nn.Identity() if cont_dim == 0 else nn.Linear(cont_dim, cont_dim)
        flat_dim = self.num_cat * embedding_dim + cont_dim
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, flat_dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(flat_dim*4, flat_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(flat_dim*2, 1)
        )

    def embed_categorical(self, x_cat):
        b = x_cat.size(0)
        shared = self.shared_emb.expand(b, self.shared_dim)
        tokens = [torch.cat([shared, emb(x_cat[:, i])], dim=1) for i, emb in enumerate(self.embs)]
        return torch.stack(tokens, dim=1)

    def forward(self, x_cat, x_cont):
        cat_tokens = self.embed_categorical(x_cat)
        micro_tokens = cat_tokens[:, list(self.micro_indices), :]
        micro_out = self.micro(micro_tokens)
        for j, idx in enumerate(self.micro_indices):
            cat_tokens[:, idx, :] = micro_out[:, j, :]
        macro_out = self.macro(cat_tokens)
        flat = macro_out.reshape(macro_out.size(0), -1)
        cont = self.cont_proj(x_cont)
        logits = self.mlp(torch.cat([flat, cont], dim=1)).squeeze(1)
        return logits



def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x_cat, x_cont, y = batch.x_cat.to(device), batch.x_cont.to(device), batch.y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cat, x_cont)
        loss = loss_fn(logits, y)
        loss.backward()
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
    auc_roc = roc_auc_score(y_all, p_all) if len(np.unique(y_all))>1 else np.nan
    prec, rec, _ = precision_recall_curve(y_all, p_all)
    auc_pr = auc_metric(rec, prec)
    preds = (p_all >= threshold).astype(int)
    rec_v = recall_score(y_all, preds)
    prec_v = precision_score(y_all, preds)
    f1_v = f1_score(y_all, preds)
    tn, fp, fn, tp = confusion_matrix(y_all, preds).ravel().tolist()
    return avg_loss, auc_roc, auc_pr, rec_v, prec_v, f1_v, (tn, fp, fn, tp)
