import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct


class InteractionBlock(nn.Module):
    def __init__(self, node_irreps="2x0e+1x1o", edge_irreps="1x0e+1x1o"):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.tensor_product = FullyConnectedTensorProduct(
            self.node_irreps, self.edge_irreps, self.node_irreps,
            internal_weights=True, shared_weights=True
        )
        self.norm = nn.LayerNorm(self.node_irreps.dim)

    def forward(self, x, edge_attr, edge_index):
        src, dst = edge_index
        print(f"x[src] shape: {x[edge_index[0]].shape}")
        # Replace None edge_attr with a tensor of ones (or zeros)
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.shape[1], device=x.device)
        print(f"edge_attr shape: {edge_attr.shape}")
        messages = self.tensor_product(x[src], edge_attr)
        return self.norm(x + messages)


class ReductionPotentialPredictor(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(100, 5)
        self.layers = nn.ModuleList([InteractionBlock() for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x = self.embedding(data.x.squeeze(-1))
        for layer in self.layers:
            x = layer(x, data.edge_attr, data.edge_index)
        pooled = global_mean_pool(x, data.batch)
        return self.fc(pooled)
