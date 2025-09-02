import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

def _check_range(x, n, name):
    mn = int(x.min().item())
    mx = int(x.max().item())
    if mn < 0 or mx >= n:
        raise RuntimeError(f"{name} out of range: min={mn}, max={mx}, allowed [0, {n-1}]")


def log_all_embeddings(writer, model, step):
    # user_ids, item_ids: list/array of raw IDs in the SAME order as embedding rows [1..N-1]
    with torch.no_grad():
        U = model.P.weight.detach().cpu()
        V = model.Q.weight.detach().cpu()

    # Projector: single-column metadata (just IDs)
    writer.add_embedding(U, metadata=[str(x) for x in range(U.shape[0])],
                         global_step=step, tag="proj/users")
    writer.add_embedding(V, metadata=[str(x) for x in range(V.shape[0])],
                         global_step=step, tag="proj/items")

def log_items_with_metadata(writer, model, meta_df, step):

    # Build rows in the same order as embeddings
    metadata_cols = ["name", "genre", "type", "episodes", "rating", "members"]
    meta_rows = meta_df[metadata_cols].astype(str).values.tolist()

    with torch.no_grad():
        V = model.Q.weight.detach().cpu()

    writer.add_embedding(
        V,
        metadata=meta_rows,
        metadata_header=metadata_cols,  # column names in Projector
        global_step=step,
        tag="proj/items_with_meta",
    )

class MF(nn.Module):
    def __init__(self, num_users, num_items, latent_size, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.num_items = num_items
        self.num_users = num_users
        self.latent_size = latent_size
        self.P = nn.Embedding(num_users, latent_size)
        self.Q = nn.Embedding(num_items, latent_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        # _check_range(user_id, self.P.num_embeddings, "user_idx")
        # _check_range(item_id, self.Q.num_embeddings, "item_idx")
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        p_bias = self.user_bias(user_id)
        q_bias = self.item_bias(item_id)
        pred_ratings = (P_u * Q_i).sum(axis = 1) + torch.squeeze(p_bias) + torch.squeeze(q_bias)
        return pred_ratings