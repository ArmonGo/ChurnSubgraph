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

