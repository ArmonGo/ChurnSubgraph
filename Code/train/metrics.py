import torch 
from sklearn.metrics import precision_recall_curve, auc
import numpy as np 
import torch 
import torch.nn.functional as F

def lift_score(y_true, y_pred, top_percent=0.1):
    """
    Calculate the Lift score at the top k% of predicted scores.

    Parameters:
    - y_true: array-like of shape (n_samples,) – true binary labels (0 or 1)
    - y_pred: array-like of shape (n_samples,) – predicted probabilities or scores
    - top_percent: float – proportion (between 0 and 1) to consider for Lift (e.g., 0.1 for top 10%)

    Returns:
    - lift: float – Lift score
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    if y_pred.dim() >1 and y_pred.shape[1] >1:
        y_pred = F.softmax(y_pred, dim = 1)
        y_pred = y_pred[:, 1] # get only class 1 prob 
    else:
        y_pred = F.sigmoid(y_pred)
    
    n = len(y_pred)
    cutoff = int(n * top_percent)
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()
    # Sort scores and select top samples
    indices = np.argsort(y_pred)[::-1]
    top_indices = indices[:cutoff]

    # Proportion of positives in top group
    top_positives = y_true[top_indices].sum() / cutoff

    # Overall proportion of positives
    base_rate = y_true.mean()

    # Lift = positive rate in top / overall positive rate
    lift = top_positives / base_rate if base_rate > 0 else np.nan
    return lift


def pr_auc_score(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    if y_pred.dim() >1 and y_pred.shape[1] >1:
        y_pred = F.softmax(y_pred, dim = 1)
        y_pred = y_pred[:, 1] # get only class 1 prob 
    else:
        y_pred = F.sigmoid(y_pred)
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    precision, recall, _ = precision_recall_curve(y_true_np, y_pred_np)
    pr_auc = auc(recall, precision)
    return  pr_auc 