from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, HeteroConv


class GraphEncoder(nn.Module):
    def __init__(self, relations, hidden=128, dropout=0.0, encoder="gat", add_self_loops=True, normalize=True):
        super().__init__()
        ent_ent = [et for et in relations if et[0] == "entity" and et[2] == "entity"]
        if not ent_ent:
            raise ValueError("No ('entity', *, 'entity') relations found.")

        self.dropout = float(dropout)
        self.encoder = str(encoder).lower()
        if self.encoder not in {"gat", "gcn"}:
            raise ValueError("encoder must be 'gat' or 'gcn'")

        if self.encoder == "gat":
            convs = {
                et: GATConv((-1, -1), hidden, heads=1, concat=False, dropout=self.dropout)
                for et in ent_ent
            }
        else:
            convs = {
                et: GCNConv(-1, hidden, add_self_loops=add_self_loops, normalize=normalize)
                for et in ent_ent
            }

        self.conv = HeteroConv(convs, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        out = {k: F.elu(v) for k, v in out.items()}
        out = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in out.items()}
        return out


class Adapter(nn.Module):
    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, x_dict, ei_dict):
        return self.enc(x_dict, ei_dict)["entity"]
