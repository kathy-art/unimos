"""GIN drug structure encoder: molecular graph (PyG Batch) -> (B, out_dim) embedding.
Uses GINEConv to incorporate bond features. Drop-in alternative to FP StructureEncoder.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool


class GINDrugEncoder(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 256,
                 out_dim: int = 256, n_layers: int = 3, dropout: float = 0.1,
                 pool: str = "mean"):
        super().__init__()
        self.atom_lin = nn.Linear(atom_dim, hidden)
        self.bond_lin = nn.Linear(bond_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.drop = nn.Dropout(dropout)
        self.pool = global_add_pool if pool == "add" else global_mean_pool
        self.out = nn.Sequential(nn.Linear(hidden, out_dim), nn.ReLU(),
                                 nn.Dropout(dropout))

    def forward(self, batch):
        x = self.atom_lin(batch.x)
        e = self.bond_lin(batch.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, batch.edge_index, e)
            x = bn(x); x = torch.relu(x); x = self.drop(x)
        g = self.pool(x, batch.batch)          # (B, hidden)
        return self.out(g)                     # (B, out_dim)


if __name__ == "__main__":
    from pathlib import Path
    from torch_geometric.data import Batch
    cache = torch.load(Path("data/processed/drug_graphs.pt"),
                       weights_only=False)
    graphs = cache["graphs"]; dims = cache["dims"]
    sample = list(graphs.values())[:8]
    b = Batch.from_data_list(sample)
    enc = GINDrugEncoder(dims["atom_dim"], dims["bond_dim"], hidden=128, out_dim=256)
    enc.eval()
    with torch.no_grad():
        out = enc(b)
    print("batch graphs:", len(sample), "-> embedding", tuple(out.shape))
    print("finite:", bool(torch.isfinite(out).all()), "| params:",
          sum(p.numel() for p in enc.parameters()))
