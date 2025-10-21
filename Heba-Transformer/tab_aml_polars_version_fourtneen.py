
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc as auc_metric,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,

)

def collate_encoded_batch(batch: List[EncodedBatch]) -> EncodedBatch:
    x_cat = torch.stack([item.x_cat for item in batch])
    if batch[0].x_cont.numel() > 0:
        x_cont = torch.stack([item.x_cont for item in batch])
    else:
        x_cont = torch.empty((len(batch), 0))
    y = torch.stack([item.y for item in batch])
    return EncodedBatch(x_cat=x_cat, x_cont=x_cont, y=y)

def fast_hash(x: str, num_buckets: int = 50000) -> int:
    """Convert any string into an integer in a fixed range [0, num_buckets)."""
    if x is None:
        return 0
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % num_buckets



# =============================================================================
# Utility functions for preprocessing and splitting
# =============================================================================

def preprocess_saml_d(df: pl.DataFrame) -> pl.DataFrame:
    """
    Feature engineering and cleaning specific to the SAML‑D dataset.
    Steps:
      1. Combines `Date` and `Time` into `timestamp`.
      2. Drops rows with null timestamps.
      3. Derives temporal features.
      4. Computes log1p of `Amount`.
      5. Regenerates a `Date` column from timestamp (for splitting).
      6. Drops redundant columns.
    """
    # Ensure Date/Time are strings; strict=False tolerates mixed types
    df = df.with_columns([
        pl.col("Date").cast(pl.Utf8, strict=False),
        pl.col("Time").cast(pl.Utf8, strict=False),
    ])

    # combine Date and Time into a single string and parse to datetime
    df = df.with_columns(
        pl.concat_str(
            [pl.col("Date").str.strip_chars(), pl.col("Time").str.strip_chars()],
            separator=" "
        )
        .str.strptime(pl.Datetime)  
        .alias("timestamp")
    )

    # Drop rows with null timestamps
    df = df.drop_nulls(["timestamp"])

    # Temporal features
    df = df.with_columns([
        pl.col("timestamp").dt.day().alias("day"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
    ])
    df = df.with_columns(
        (pl.col("day_of_week") >= 5).cast(pl.Int32).alias("is_weekend")
    )

    # Log-transform Amount safely (handles nulls)
    df = df.with_columns(
        pl.when(pl.col("Amount").is_not_null())
        .then(pl.col("Amount").cast(pl.Float64).log1p())
        .otherwise(None)
        .alias("amount_log")
    )

    # Regenerate pure Date for calendar-based split
    df = df.with_columns(pl.col("timestamp").dt.date().alias("Date"))

    # Drop redundant columns
    df = df.drop(["Laundering_type", "Time", "Amount"], strict=False)
    return df


def custom_split_polars(df: pl.DataFrame, validation_dt: int = 70, test_dt: int = 35):
    """
    Split a Polars DataFrame into train/validation/test based on calendar-day cutoffs.
    Uses the maximum date in the DataFrame to compute cutoffs backwards.
    """
    # Ensure Date column is datetime; parse if needed
    try:
        max_date = df.select(pl.col("Date").max()).to_series()[0]
    except Exception:
        df = df.with_columns(
            pl.col("Date").str.strptime(pl.Datetime, fmt=None, strict=False).alias("Date")
        )
        max_date = df.select(pl.col("Date").max()).to_series()[0]

    # Compute cutoffs
    test_cutoff = max_date - timedelta(days=test_dt)
    validation_cutoff = max_date - timedelta(days=validation_dt)

    # Filter sets
    test_set = df.filter(pl.col("Date") >= pl.lit(test_cutoff))
    validation_set = df.filter(
        (pl.col("Date") >= pl.lit(validation_cutoff)) & (pl.col("Date") < pl.lit(test_cutoff))
    )
    train_set = df.filter(pl.col("Date") < pl.lit(validation_cutoff))

    return train_set, validation_set, test_set


def recast(df: pl.DataFrame) -> pl.DataFrame:
    """
    Downcast integer columns to smaller dtypes to save memory.
    Excludes 'Sender_account' and 'Receiver_account'.
    """
    exclude = ['Sender_account', 'Receiver_account']
    for col in df.columns:
        if col not in exclude:
            dtype = df[col].dtype
            if dtype in (pl.Int64, pl.Int32):
                # Drop nulls to get correct max
                maxval = df[col].drop_nulls().max()
                if maxval is not None:
                    if maxval < 127:
                        df = df.with_columns(pl.col(col).cast(pl.Int8))
                    elif maxval < 32767:
                        df = df.with_columns(pl.col(col).cast(pl.Int16))
                    elif maxval < 2147483647:
                        df = df.with_columns(pl.col(col).cast(pl.Int32))
    return df

# =============================================================================
# Dataset and DataLoader utilities
# =============================================================================

@dataclass
class EncodedBatch:
    """
    Simple container for batches returned by the dataset.
    """
    x_cat: torch.LongTensor
    x_cont: torch.FloatTensor
    y: torch.FloatTensor


class TabAMLDataset(Dataset):
    """
    Handles preprocessing of categorical and continuous features stored in a Polars DataFrame.
    Encodes categorical columns with LabelEncoder and scales continuous columns.
    Supports both training (fit=True) and inference (fit=False) modes.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        cat_cols: List[str],
        cont_cols: List[str],
        label_col: str,
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        scaler: Optional[StandardScaler] = None,
        fit: bool = True,
    ):
        super().__init__()
        self.df = df.clone()
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_col = label_col

        # Convert labels to float32 numpy array
        self.y = self.df[label_col].cast(pl.Float32).to_numpy()

        # Use provided encoders or initialize new ones
        self.encoders: Dict[str, LabelEncoder] = encoders or {}
        encoded_cols: List[np.ndarray] = []

        for col in self.cat_cols:
            col_values = self.df[col].cast(pl.Utf8).to_list()

            #Hash high-cardinality columns
            if col in ["Sender_account", "Receiver_account"]:
                print(f"⚡ Using hashing for high-cardinality column: {col}")
                hashed = np.array([fast_hash(v) for v in col_values], dtype=np.int64)
                encoded_cols.append(hashed)
                continue

            #Standard label encoding for normal categorical features
            if fit or col not in self.encoders:
                le = LabelEncoder()
                le.fit(col_values + ["UNK"])
                self.encoders[col] = le
            else:
                le = self.encoders[col]

            #Replace unseen values with "UNK"
            mapped = [v if v in le.classes_ else "UNK" for v in col_values]
            encoded_cols.append(le.transform(mapped))

        #Stack categorical columns (after all encoding)
        if encoded_cols:
            self.x_cat = np.stack(encoded_cols, axis=1).astype(np.int64)
        else:
            self.x_cat = np.empty((len(self.df), 0), dtype=np.int64)
        print(f"[DEBUG] x_cat shape: {self.x_cat.shape}")
        #Scale continuous columns
        if self.cont_cols:
            cont_array = self.df.select(self.cont_cols).to_numpy()
            if fit or scaler is None:
                self.scaler = StandardScaler()
                self.x_cont = self.scaler.fit_transform(cont_array)
            else:
                self.scaler = scaler
                self.x_cont = self.scaler.transform(cont_array)
        else:
            self.scaler = None
            self.x_cont = np.empty((len(self.df), 0), dtype=np.float32)

        #Category sizes for embedding layers
        self.category_sizes = [
            (len(self.encoders[col].classes_) if col in self.encoders else 50000)
            for col in self.cat_cols
        ]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> EncodedBatch:
        cat = torch.from_numpy(self.x_cat[idx]).long()
        cont = torch.from_numpy(self.x_cont[idx]).float() if self.x_cont.size > 0 else torch.empty(0)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return EncodedBatch(x_cat=cat, x_cont=cont, y=label)



def build_weighted_sampler(labels: pl.Series) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to oversample minority class.
    """
    counts_df = labels.value_counts()
    counts = {row[0]: row[1] for row in counts_df.to_numpy()}
    weights = [1.0 / counts[x] for x in labels.to_list()]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_pos_weight(labels: pl.Series) -> float:
    """
    Compute positive class weight for BCEWithLogitsLoss.
    """
    pos = labels.sum()
    neg = len(labels) - pos
    return float(neg / pos) if pos > 0 else 1.0


# =============================================================================
# Tab-AML Model components (same as original transformer)
# =============================================================================

class ResidualAttentionLayer(nn.Module):
    """
    A transformer encoder layer with residual attention.
    See original Tab-AML implementation.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or d_model * 4
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = (d_model // nhead) ** -0.5

    def forward(self, x: torch.Tensor, prev_attn: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        h = self.nhead

        # Layer norm
        x_norm = self.norm1(x)

        # Compute Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)

        # Raw attention scores
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if prev_attn is not None:
            attn_scores = attn_scores + prev_attn

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)

        out = self.attn_out(out)
        x = x + out

        x_ff = self.norm2(x)
        x_ff = self.ffn(x_ff)
        x = x + x_ff

        return x, attn_scores.detach()


class ResidualAttentionEncoder(nn.Module):
    """Stack of ResidualAttentionLayer modules."""
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_attn: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, prev_attn = layer(x, prev_attn)
        return x


class TabAMLModel(nn.Module):
    """
    Main Tab‑AML transformer.
    Embeds categorical features (shared + individual), uses micro and macro encoders,
    concatenates flattened output with continuous features, and feeds through MLP.
    """
    def __init__(
        self,
        category_sizes: List[int],
        cont_dim: int,
        embedding_dim: int = 64,
        shared_ratio: float = 1 / 6,
        num_heads: int = 4,
        num_layers1: int = 2,
        num_layers2: int = 2,
        dropout: float = 0.25,
        micro_indices: Tuple[int, int] = (0, 1),
    ):
        super().__init__()
        assert 0 < shared_ratio < 1
        self.num_cat = len(category_sizes)
        self.cont_dim = cont_dim
        self.embedding_dim = embedding_dim
        self.shared_dim = max(1, int(embedding_dim * shared_ratio))
        self.indiv_dim = embedding_dim - self.shared_dim
        self.micro_indices = micro_indices

        self.shared_embedding = nn.Parameter(torch.randn(self.shared_dim))
        self.embeddings = nn.ModuleList([nn.Embedding(sz, self.indiv_dim) for sz in category_sizes])

        self.micro_encoder = ResidualAttentionEncoder(
            num_layers1, d_model=embedding_dim, nhead=num_heads, dropout=dropout
        )
        self.macro_encoder = ResidualAttentionEncoder(
            num_layers2, d_model=embedding_dim, nhead=num_heads, dropout=dropout
        )

        self.cont_proj = nn.Identity() if cont_dim == 0 else nn.Linear(cont_dim, cont_dim)

        flattened_cat_dim = self.num_cat * embedding_dim
        mlp_input_dim = flattened_cat_dim + cont_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim * 4, mlp_input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim * 2, 1),
        )

    def embed_categorical(self, x_cat: torch.LongTensor) -> torch.Tensor:
        batch_size = x_cat.size(0)
        shared = self.shared_embedding.expand(batch_size, self.shared_dim)
        tokens = []
        for i, emb in enumerate(self.embeddings):
            indiv = emb(x_cat[:, i])
            token = torch.cat([shared, indiv], dim=1)
            tokens.append(token)
        return torch.stack(tokens, dim=1)

    def forward(self, x_cat: torch.LongTensor, x_cont: torch.FloatTensor) -> torch.Tensor:
        cat_tokens = self.embed_categorical(x_cat)

        # Micro encoder on selected indices
        if max(self.micro_indices) >= self.num_cat:
            raise IndexError(f"micro_indices {self.micro_indices} out of range")
        micro_tokens = cat_tokens[:, list(self.micro_indices), :]
        micro_out = self.micro_encoder(micro_tokens)
        cat_tokens = cat_tokens.clone()
        for idx, token_idx in enumerate(self.micro_indices):
            cat_tokens[:, token_idx, :] = micro_out[:, idx, :]

        # Macro encoder over all tokens
        macro_out = self.macro_encoder(cat_tokens)

        b, n, d = macro_out.shape
        flat_cat = macro_out.reshape(b, n * d)
        if self.cont_dim > 0:
            cont = self.cont_proj(x_cont)
            x = torch.cat([flat_cat, cont], dim=1)
        else:
            x = flat_cat
        logits = self.mlp(x).squeeze(1)
        return logits


# =============================================================================
# Training and evaluation routines (adapted from original)
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
) -> float:
    model.train()
    device = device or next(model.parameters()).device
    total_loss = 0.0
    count = 0
    for batch in loader:
        x_cat = batch.x_cat.to(device)
        x_cont = batch.x_cont.to(device)
        labels = batch.y.to(device)

        optimizer.zero_grad()
        logits = model(x_cat, x_cont)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        count += labels.size(0)
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float, float, float, Tuple[int, int, int, int]]:
    """
    Evaluate the model and return:
      - loss
      - ROC-AUC
      - PR-AUC
      - recall, precision, F1
      - confusion matrix (TN, FP, FN, TP)
    """
    model.eval()
    device = device or next(model.parameters()).device
    total_loss = 0.0
    count = 0
    labels_list: List[float] = []
    probs_list: List[float] = []
    for batch in loader:
        x_cat = batch.x_cat.to(device)
        x_cont = batch.x_cont.to(device)
        labels = batch.y.to(device)
        logits = model(x_cat, x_cont)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        count += labels.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_list.extend(labels.cpu().numpy())
        probs_list.extend(probs)

    avg_loss = total_loss / max(count, 1)

    try:
        auc_roc = roc_auc_score(labels_list, probs_list)
    except ValueError:
        auc_roc = float("nan")

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(labels_list, probs_list)
        auc_pr = auc_metric(recall_curve, precision_curve)
    except ValueError:
        auc_pr = float("nan")

    preds = (np.array(probs_list) >= threshold).astype(int)
    try:
        recall_val = recall_score(labels_list, preds)
        precision_val = precision_score(labels_list, preds)
        f1_val = f1_score(labels_list, preds)
        tn, fp, fn, tp = confusion_matrix(labels_list, preds).ravel().tolist()
    except Exception:
        recall_val = precision_val = f1_val = float("nan")
        tn = fp = fn = tp = 0

    return avg_loss, auc_roc, auc_pr, recall_val, precision_val, f1_val, (tn, fp, fn, tp)

