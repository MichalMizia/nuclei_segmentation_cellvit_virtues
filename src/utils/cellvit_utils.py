import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_dice_score(preds, targets, num_classes, ignore_index=None):
    dice_scores = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
            
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = pred_mask.float().sum() + target_mask.float().sum()
        
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
        
    return np.mean(dice_scores)

def calculate_f1_score(preds, targets, num_classes, ignore_index=None):
    return calculate_dice_score(preds, targets, num_classes, ignore_index)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, softmax=True, ignore_index=None, weight=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.softmax = softmax
        self.ignore_index = ignore_index
        self.weight = weight # Expects Tensor of shape (num_classes,)
        self.smooth = smooth

    def forward(self, logits, targets):
        if self.softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits
            
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
            
            # Smooth added to numerator and denominator
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            class_loss = (1.0 - dice).mean()
            
            # Apply class weight if provided
            if self.weight is not None:
                class_loss = class_loss * self.weight[cls]
                
            loss += class_loss
            count += 1
            
        return loss / count

class CombinedLoss(nn.Module):
    """
    Use this for: nuclei_binary_map (2 classes) and nuclei_type_map (N classes)
    """
    def __init__(self, num_classes, alpha=0.5, ignore_index=None, class_weights=None):
        super().__init__()
        self.alpha = alpha
        
        # Ensure weights are on the correct device when used
        self.class_weights = class_weights
        
        self.ce = nn.CrossEntropyLoss(
            weight=self.class_weights, 
            ignore_index=ignore_index if ignore_index is not None else -100
        )
        # Pass weights to Dice as well
        self.dice = DiceLoss(num_classes, ignore_index=ignore_index, weight=self.class_weights)
        
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets.long())
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

class HVLoss(nn.Module):
    """
    Use this ONLY for the 'hv_map' branch.
    HV map is a regression task (gradients), not classification.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # targets for HV map should be float, not long
        return self.mse(preds, targets.float())