# source: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py
"""
@inproceedings{
    Ren2020balms,
    title={Balanced Meta-Softmax for Long-Tailed Visual Recognition},
    author={Jiawei Ren and Cunjun Yu and Shunan Sheng and Xiao Ma and Haiyu Zhao and Shuai Yi and Hongsheng Li},
    booktitle={Proceedings of Neural Information Processing Systems(NeurIPS)},
    month = {Dec},
    year={2020}
}

[1] Lin, T. Y., et al. "Focal loss for dense object detection." arXiv 2017." arXiv preprint arXiv:1708.02002 (2002).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
from focal_loss.focal_loss  import FocalLoss

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def register_log_weights(self, count_neg, count_pos):
        self.register_buffer("log_class_counts", torch.log(torch.tensor([count_neg, count_pos], dtype=torch.float)))


    def forward(self, logits, target):
        """
        logits: Tensor [batch_size, num_classes] (raw logits from model)
        target: Tensor [batch_size] (ground truth class indices)
        """
        count_pos = target.sum() 
        count_neg = len(target) - count_pos
        self.register_log_weights(count_neg, count_pos)
        self.log_class_counts = self.log_class_counts.to(logits.device)
        # Shift logits by log class prior
        balanced_logits = logits + self.log_class_counts
        # Compute cross-entropy loss on shifted logits
        return F.cross_entropy(balanced_logits, target.long())
        
    

def focal_loss(logits, targets, alpha, gamma):
    m = torch.nn.Sigmoid()
    loss_fn = FocalLoss(gamma = gamma, weights =  torch.FloatTensor([1, alpha]).to(targets.device)) # class 0 weight equal to 1 by default 
    return loss_fn(m(logits.squeeze()), targets.long())

