import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_dice_score(preds, targets, num_classes, ignore_index=None):
    """
    Calculate Dice Score for multi-class segmentation.
    
    Args:
        preds (torch.Tensor): Predictions (B, H, W) with class indices.
        targets (torch.Tensor): Ground truth (B, H, W) with class indices.
        num_classes (int): Number of classes.
        ignore_index (int, optional): Index to ignore.
        
    Returns:
        float: Mean Dice Score across classes (macro-average).
    """
    dice_scores = []
    
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
            
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = pred_mask.float().sum() + target_mask.float().sum()
        
        if union == 0:
            dice_scores.append(1.0) # If both are empty, it's a match
        else:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
        
    return np.mean(dice_scores)

def calculate_f1_score(preds, targets, num_classes, ignore_index=None):
    """
    Calculate F1 Score (which is essentially Dice Score for binary, but here for multi-class).
    F1 = 2 * (precision * recall) / (precision + recall)
    In segmentation, F1 is often synonymous with Dice.
    """
    return calculate_dice_score(preds, targets, num_classes, ignore_index)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, softmax=True, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax = softmax
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (B, C, H, W)
            targets (torch.Tensor): (B, H, W)
        """
        if self.softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits
            
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        loss = 0.0
        count = 0
        
        for cls in range(self.num_classes):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue
            
            p = probs[:, cls, :, :]
            t = targets_one_hot[:, cls, :, :]
            
            intersection = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
            loss += (1.0 - dice).mean()
            count += 1
            
        return loss / count

class CombinedLoss(nn.Module):
    """
    Combined Cross Entropy and Dice Loss.
    """
    def __init__(self, num_classes, alpha=0.5, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index is not None else -100)
        self.dice = DiceLoss(num_classes, ignore_index=ignore_index)
        
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets.long())
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
