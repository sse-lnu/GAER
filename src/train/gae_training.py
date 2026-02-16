from __future__ import annotations

import torch
from torch_geometric.nn import GAE
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from models.gnn_encoder import Adapter, GraphEncoder


def merge_entity_edges(edge_index_dict, num_nodes, device):
    parts = [ei for (et, ei) in edge_index_dict.items() if et[0] == "entity" and et[2] == "entity"]
    if not parts:
        raise ValueError("No ('entity', *, 'entity') edges present.")

    ei = torch.cat(parts, dim=1)
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei = coalesce(ei, num_nodes=num_nodes)
    return ei.to(device)


def train(
    data,
    hidden=64,
    dropout=0.0,
    epochs=30,
    lr=1e-3,
    encoder="gat",
    device=None,
    add_self_loops=True,
    normalize=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_dict = {"entity": data["entity"].x.to(device)}
    ei_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    pos_edge_index = merge_entity_edges(ei_dict, data["entity"].num_nodes, device)

    enc = GraphEncoder(
        relations=data.edge_types,
        hidden=hidden,
        dropout=dropout,
        encoder=encoder,
        add_self_loops=add_self_loops,
        normalize=normalize,
    ).to(device)

    model = GAE(Adapter(enc)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = None
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model.encode(x_dict, ei_dict)
        loss = model.recon_loss(z, pos_edge_index)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        z = model.encode(x_dict, ei_dict).detach().cpu().numpy()

    return z, {"loss": last_loss}
