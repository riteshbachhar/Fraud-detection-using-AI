from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

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


@dataclass
class EncodedBatch:
    # Simple container for batches returned by the dataset.
    x_cat: torch.LongTensor
    x_cont: torch.FloatTensor
    y: torch.FloatTensor


class TabAMLDataset(Dataset):
   
    #Handles preprocessing of categorical and continuous features.  It encodes categorical variables using ``LabelEncoder`` and normalises continuous variables using ``StandardScaler``.  Both encoders are
    #stored on the dataset instance for potential reuse or inspection.

    

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        cont_cols: List[str],
        label_col: str,
    ) -> None:
        super().__init__()
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.label_col = label_col

        
        # Convert label to float tensor
        self.y = df[self.label_col].astype("float32").values

        # Fit label encoders for each categorical column.  Any unseen
        # category at inference time will be mapped to index 0 (UNK).
        self.encoders: Dict[str, LabelEncoder] = {}
        encoded = []
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(list(df[col]) + ["UNK"])
            self.encoders[col] = le
            encoded.append(le.transform(df[col]))
        # Shape: (num_samples, num_categorical_features)
        self.x_cat = np.stack(encoded, axis=1).astype("int64")

        # Fit scaler for continuous columns.  If there are no
        # continuous features, create an empty array for x_cont. In our case, we have but just if someone needs to use it on different dataset.
        if self.cont_cols:
            self.scaler = StandardScaler()
            self.x_cont = self.scaler.fit_transform(df[self.cont_cols])
        else:
            self.scaler = None
            self.x_cont = np.empty((len(df), 0), dtype="float32")

        # Precompute category sizes (unique values + 1 for unknown)
        self.category_sizes = [len(self.encoders[col].classes_) for col in self.cat_cols]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> EncodedBatch:
        # Extract categorical and continuous features for a single sample
        cat = torch.from_numpy(self.x_cat[idx]).long()
        cont = (
            torch.from_numpy(self.x_cont[idx]).float()
            if self.x_cont.size > 0
            else torch.empty(0)
        )
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return EncodedBatch(x_cat=cat, x_cont=cont, y=label)


class ResidualAttentionLayer(nn.Module):
    """
    A transformer encoder layer with residual attention.  Unlike the
    standard PyTorch ``TransformerEncoderLayer``, this layer retains
    the attention scores from the previous layer and adds them to
    the raw similarity matrix before softmax.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: Optional[int] = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or d_model * 4
        self.dropout = dropout

        # Projection layers for Q, K, V
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalisations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

        # Scale factor for dot product attention
        self.scale = (d_model // nhead) ** -0.5

    def forward(self, x: torch.Tensor, prev_attn: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        h = self.nhead

        # Layer norm on input
        x_norm = self.norm1(x)

        # Compute Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x_norm)  # (b, t, 3*d)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)

        # Compute raw attention scores
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Add residual attention from previous layer if provided
        if prev_attn is not None:
            attn_scores = attn_scores + prev_attn

        # Softmax to obtain attention weights
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)

        # Project back to d_model
        out = self.attn_out(out)

        # Residual connection for attention
        x = x + out

        # Feedforward sublayer with residual connection
        x_ff = self.norm2(x)
        x_ff = self.ffn(x_ff)
        x = x + x_ff

        return x, attn_scores.detach()


class ResidualAttentionEncoder(nn.Module):
    #A stack of ``ResidualAttentionLayer`` modules

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_attn: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, prev_attn = layer(x, prev_attn)
        return x


class TabAMLModel(nn.Module):
    """
    The main model. This model  embeds categorical features using
    individual and shared embeddings, applies a two‑stage residual
    attention encoder (micro and macro), and concatenates the
    flattened output with scaled continuous features.  A multi‑layer
    perceptron (MLP) then produces a binary prediction.
    """

    def __init__(
        self,
        category_sizes: List[int],
        cont_dim: int,
        embedding_dim: int = 32,
        shared_ratio: float = 1 / 8,
        num_heads: int = 4,
        num_layers1: int = 1,
        num_layers2: int = 1,
        dropout: float = 0.1,
        micro_indices: Tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        assert 0 < shared_ratio < 1
        self.num_cat = len(category_sizes)
        self.cont_dim = cont_dim
        self.embedding_dim = embedding_dim
        self.shared_dim = max(1, int(embedding_dim * shared_ratio))
        self.indiv_dim = embedding_dim - self.shared_dim
        self.micro_indices = micro_indices

        # Shared embedding vector
        self.shared_embedding = nn.Parameter(torch.randn(self.shared_dim))

        # Individual embeddings for each categorical feature
        self.embeddings = nn.ModuleList(
            [nn.Embedding(sz, self.indiv_dim) for sz in category_sizes]
        )

        # Micro and macro encoders
        self.micro_encoder = ResidualAttentionEncoder(
            num_layers1, d_model=embedding_dim, nhead=num_heads, dropout=dropout
        )
        self.macro_encoder = ResidualAttentionEncoder(
            num_layers2, d_model=embedding_dim, nhead=num_heads, dropout=dropout
        )

        # Continuous projection (identity if no continuous features)
        self.cont_proj = nn.Identity() if cont_dim == 0 else nn.Linear(cont_dim, cont_dim)

        # Final MLP classifier
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
        # Embed categorical features
        cat_tokens = self.embed_categorical(x_cat)

        # Micro encoder on selected indices
        if max(self.micro_indices) >= self.num_cat:
            raise IndexError(
                f"micro_indices {self.micro_indices} refer to positions outside range of categorical features"
            )
        micro_tokens = cat_tokens[:, list(self.micro_indices), :]
        micro_out = self.micro_encoder(micro_tokens)
        cat_tokens = cat_tokens.clone()
        for idx, token_idx in enumerate(self.micro_indices):
            cat_tokens[:, token_idx, :] = micro_out[:, idx, :]

        # Macro encoder over all tokens
        macro_out = self.macro_encoder(cat_tokens)

        # Flatten and concatenate with continuous features
        b, n, d = macro_out.shape
        flat_cat = macro_out.reshape(b, n * d)
        if self.cont_dim > 0:
            cont = self.cont_proj(x_cont)
            x = torch.cat([flat_cat, cont], dim=1)
        else:
            x = flat_cat
        logits = self.mlp(x).squeeze(1)
        return logits


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
) -> float:
    # Train the model for a single epoch and return average loss.
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
    Evaluate the model on a validation or test set.  Returns a tuple
    containing (loss, ROC‑AUC, PR‑AUC, recall, precision, F1, confusion
    matrix counts) where confusion matrix counts are (TN, FP, FN, TP)
    based on the provided threshold.
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

    # Derive predictions at specified threshold
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


def preprocess_saml_d(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering and cleaning
    # combine date + time into timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])

    # temporal features
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # log-transform amount
    df["amount_log"] = np.log1p(df["Amount"])

    # remove redundant fields
    df = df.drop(columns=["Laundering_type", "Date", "Time", "Amount"], errors="ignore")

    return df


def chronological_split(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    #Split the dataframe based on chronological order (80% train, 10% validation, 10% test).

    
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    return train, val, test


def compute_pos_weight(labels: pd.Series) -> float:
    """
    Compute a positive class weight for BCEWithLogitsLoss based on
    the ratio of negative to positive samples.  If there are no
    positive samples, return 1.0.
    """
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0:
        return 1.0
    return float(neg / pos)


def build_weighted_sampler(labels: pd.Series) -> WeightedRandomSampler:
    """
    Construct a WeightedRandomSampler that oversamples the minority
    class.  Each sample's weight is the inverse of its class
    frequency.
    """
    class_counts = labels.value_counts().to_dict()
    weights = labels.map(lambda x: 1.0 / class_counts[x]).tolist()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def collate_encoded_batch(batch: List[EncodedBatch]) -> EncodedBatch:
    #Custom collate function for batching ``EncodedBatch`` objects
    x_cat = torch.stack([item.x_cat for item in batch])
    if batch[0].x_cont.numel() > 0:
        x_cont = torch.stack([item.x_cont for item in batch])
    else:
        x_cont = torch.empty((len(batch), 0))
    y = torch.stack([item.y for item in batch])
    return EncodedBatch(x_cat=x_cat, x_cont=x_cont, y=y)


