import torch
import torch.nn as nn
import torch.nn.functional as F

## Imbalanced graph learning via mixed entropy minimization
## https://github.com/12chen20/GraphME
"""
citation: 
@article{xu2024imbalanced,
	title        = {Imbalanced graph learning via mixed entropy minimization},
	author       = {Xu, Liwen and Zhu, Huaguang and Chen, Jiali},
	year         = 2024,
	journal      = {Scientific Reports},
	publisher    = {Nature Publishing Group UK London},
	volume       = 14,
	number       = 1,
	pages        = 24892,
	doi          = {10.1038/s41598-024-75999-6}
}
"""

class ME(nn.Module):
    def __init__(self, balancing_factor=0.3):
        super().__init__()
        self.nll_loss = nn.NLLLoss()
        # self.device = device # {'cpu', 'cuda:0', 'cuda:1', ...}
        self.balancing_factor = balancing_factor

    def forward(self, yHat, y):
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y.long())
        # regularization entropy
        yHat = F.softmax(yHat, dim=1)
        #Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        Px = yHat # / (1 - Yg + 1e-7)
        Px_log = torch.log(Px + 1e-10)
    
        output = Px * Px_log
        regularization_entropy = torch.sum(output) / (float(batch_size * float(yHat.shape[1])))

        return cross_entropy - self.balancing_factor * regularization_entropy
    
