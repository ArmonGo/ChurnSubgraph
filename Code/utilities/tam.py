## TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification
## https://github.com/Jaeyun-Song/TAM/blob/main/models/pc_softmax.py

"""
@InProceedings{pmlr-v162-song22a,
  title = 	 {{TAM}: Topology-Aware Margin Loss for Class-Imbalanced Node Classification},
  author =       {Song, Jaeyun and Park, Joonhyung and Yang, Eunho},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {20369--20383},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
}

"""


from __future__ import annotations

from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

__all__: Sequence[str] = [
    "compute_jsd",
    "compute_tam",
    "adjust_output",
    "MultiHeadEdgeAggregation",
    "MeanAggregation",  # backward‑compat alias
]

# ---------------------------------------------------------------------------
#                  Divergence helper 
# ---------------------------------------------------------------------------

def compute_jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    jsd = 0.5 * (
        F.kl_div(m.log(), p, reduction="none") + F.kl_div(m.log(), q, reduction="none")
    )
    return jsd


# ---------------------------------------------------------------------------
#              Edge‑aware neighbourhood mean with learnable weights
# different from the original paper as i add learnable weights here, but the old paper just use mean aggregated message 
# ---------------------------------------------------------------------------
class MultiHeadEdgeAggregation(MessagePassing):
    """Mean aggregator that learns a **scalar weight** from multi‑dim edge attr.

    Parameters
    ----------
    edge_dim   : int            – dimensionality of ``edge_attr``.
    hidden_dim : int, default=32 – width of the MLP hidden layer.
    normalize  : bool, default=True – divide by *weighted* degree.
    act        : str, default="softplus" – final activation ("softplus" or
                                        "sigmoid").
    """

    def __init__(
        self,
        edge_dim: int,
        *,
        hidden_dim: int = 32,
        normalize: bool = True,
        act: str = "softplus",
    ):
        super().__init__(aggr="add")
        self._normalize = normalize

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if act == "sigmoid":
            self.act: nn.Module = nn.Sigmoid()
        elif act == "softplus":
            self.act = nn.Softplus()
        else:
            raise ValueError("Unsupported activation: choose 'softplus' or 'sigmoid'.")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # noqa: D401
        num_nodes = x.size(0)
        if edge_attr is None:
            # fall back to scalar 1.0 per edge
            edge_attr = x.new_zeros(edge_index.size(1), 1)
        elif edge_attr.dim() == 1:
            # promote scalar → vector
            edge_attr = edge_attr.unsqueeze(-1)

        # --------------------------------- add self‑loops -------------------
        loop_attr = x.new_zeros(num_nodes, edge_attr.size(-1))
        edge_index, edge_attr = add_self_loops(
            edge_index,
            edge_attr=edge_attr,
            fill_value="add",
            num_nodes=num_nodes
        )
        edge_weight = self.act(self.mlp(edge_attr)).squeeze(-1)  # [E]

        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, size=(num_nodes, num_nodes)
        )

        if self._normalize:
            deg = scatter_add(edge_weight, edge_index[1], dim=0, dim_size=num_nodes).clamp_min(1e-12)
            out = out / deg.unsqueeze(-1)
        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:  # noqa: N802
        return edge_weight.unsqueeze(-1) * x_j



class MeanAggregation(MessagePassing):
    def __init__(self):
        super(MeanAggregation, self).__init__(aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        _edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(_edge_index, x=x)


# ---------------------------------------------------------------------------
#                       Main TAM (unchanged except var names)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_tam(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    label: torch.Tensor,
    train_mask: torch.Tensor,
    aggregator: MessagePassing,
    *,
    class_num_list: Sequence[int] | torch.Tensor,
    temp_phi: float = 2.0,
    temp_gamma: float = 0.4,
    edge_attr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = logits.device
    n_cls = int(label.max().item()) + 1

    cls_counts = torch.as_tensor(class_num_list, dtype=torch.float, device=device)
    cls_ratio = cls_counts / cls_counts.sum()
    cls_ratio = cls_ratio * temp_gamma + (1.0 - temp_gamma)
    max_beta = cls_ratio.max()
    cls_temperature = (temp_phi * (cls_ratio + 1 - max_beta)).unsqueeze(0)  # [1, C]
    inv_temp = 1.0 / cls_temperature

    # get the adjusted prob
    prob = F.softmax(logits.detach() * inv_temp, dim=1)
    prob[train_mask] = F.one_hot(label[train_mask], num_classes=n_cls).float()

    neighbor_dist = aggregator(prob, edge_index, edge_attr=edge_attr)[train_mask]
   
    # Compute class-wise connectivity matrix
    conn = torch.stack([
        neighbor_dist[label[train_mask] == c].mean(dim=0) for c in range(n_cls)
    ])
    
    # Preprocess class-wise connectivity matrix and NLD for numerical stability
    center_mask = F.one_hot(label[train_mask], num_classes=n_cls).bool()
    neighbor_dist = neighbor_dist.clamp_min(1e-6)
    conn = conn.clamp_min(1e-6)

    # Compute ACM
    diag_conn = torch.diagonal(conn)
    acm = (
        neighbor_dist[center_mask].unsqueeze(1)
        / diag_conn[label[train_mask]].unsqueeze(1)
    ) * (conn[label[train_mask]] / neighbor_dist)
    acm = acm.clamp_max(1.0)
    acm[center_mask] = 1.0

    cls_pair_jsd = compute_jsd(conn.unsqueeze(0), conn.unsqueeze(1)).sum(dim=-1).clamp_min(1e-6)
    self_kl = compute_jsd(neighbor_dist, conn[label[train_mask]]).sum(dim=-1, keepdim=True)
    neighbor_kl = compute_jsd(neighbor_dist.unsqueeze(1), conn.unsqueeze(0)).sum(dim=-1)

    adm = (
        self_kl.pow(2)
        + cls_pair_jsd[label[train_mask]].pow(2)
        - neighbor_kl.pow(2)
    ) / (2.0 * cls_pair_jsd[label[train_mask]].pow(2))
    adm[center_mask] = 0.0
    return acm, adm

# ---------------------------------------------------------------------------
#                         Logit adjustment helper
# ---------------------------------------------------------------------------

def adjust_output(
    args,
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    label: torch.Tensor,
    train_mask: torch.Tensor,
    aggregator: MessagePassing,
    class_num_list: Sequence[int],
    warmup: bool, 
    *,
    edge_attr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if args.tam and not warmup:
        acm, adm = compute_tam(
            logits,
            edge_index,
            label,
            train_mask,
            aggregator,
            class_num_list=class_num_list,
            temp_phi=args.temp_phi,
            temp_gamma=args.temp_gamma,
            edge_attr=edge_attr,
        )
        logits_adj = logits[train_mask] + args.tam_alpha * acm.log() - args.tam_beta * adm
        return logits_adj
    return logits[train_mask]

