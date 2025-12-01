import torch.nn.functional as F
import torch 
from sklearn.metrics import precision_recall_curve, auc


def cross_entropy_with_logits(logits, targets):
    return F.cross_entropy(logits, targets.long())

def neg_pr_auc_pytorch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute negative precision-recall AUC.
    Inputs:
        y_pred: Tensor of predicted probabilities (float), shape (n,)
        y_true: Tensor of ground-truth labels (0 or 1), shape (n,)
    Output:
        Negative PR-AUC (float), for Optuna minimization
    """
    # Move to CPU and convert to numpy
    y_pred = y_pred.squeeze()
    if y_pred.dim() >1:
        y_pred = F.softmax(y_pred, dim = 1)
        y_pred = y_pred[:, 1] # get only class 1 prob 
    else:
        y_pred = F.sigmoid(y_pred)
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    precision, recall, _ = precision_recall_curve(y_true_np, y_pred_np)
    pr_auc = auc(recall, precision)
    return torch.tensor([-pr_auc]) # make the torch output


def sample_pos_neg(y, num_pos, num_neg_per_pos):
    """
    y:         tensor of shape [N], binary labels
    mask:      boolean mask selecting which nodes to sample from
    num_pos:   number of positive samples (P)
    num_neg_per_pos: number of negatives per positive (K)
    
    Returns:
        pos_idx: shape [P]
        neg_idx: shape [P, K]
    """
    # Valid nodes according to mask
    valid_idx = torch.tensor(range(len(y)))
    pos_pool = valid_idx[y.detach().cpu() == 1]
    neg_pool = valid_idx[y.detach().cpu() == 0]

    # Sample P positives
    pos_idx = pos_pool[torch.randint(0, len(pos_pool), (num_pos,))]

    # Sample P*K negatives
    K = num_neg_per_pos
    neg_idx = neg_pool[torch.randint(0, len(neg_pool), (num_pos * K,))]
    neg_idx = neg_idx.view(num_pos, K)   # reshape → [P, K]

    return pos_idx, neg_idx

def bpr_loss_from_indices(y_pred, pos_idx, neg_idx):
    """
    y_pred: [N] scores from model
    pos_idx: [P]
    neg_idx: [P, K]
    """
    pos_scores = y_pred[pos_idx]               # shape [P]
    neg_scores = y_pred[neg_idx]               # shape [P, K]

    # BPR loss: maximize pos - neg
    loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores))
    return loss.mean()

def compute_bpr_loss(y_pred, y, num_pos=32, neg_per_pos=3):
    # num_pos use all 
    num_pos = max(int(y.sum().detach().item()), num_pos)
    pos_idx, neg_idx = sample_pos_neg(
        y=y,
        num_pos=num_pos,
        num_neg_per_pos=neg_per_pos
    )
    
    return bpr_loss_from_indices(y_pred, pos_idx, neg_idx)

