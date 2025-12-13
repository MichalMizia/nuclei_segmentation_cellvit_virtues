from sympy import beta
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation


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

        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).float().sum()
        union = pred_mask.float().sum() + target_mask.float().sum()

        if union == 0:
            dice_scores.append(1.0)  # If both are empty, it's a match
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


def calculate_per_class_metrics(preds, targets, num_classes, class_names=None):
    """
    Calculate Dice, IoU, Precision, Recall per class.
    Helps identify which cell types are failing.
    """
    metrics = {}

    for cls in range(num_classes):
        pred_mask = (preds == cls).float()
        target_mask = (targets == cls).float()

        TP = (pred_mask * target_mask).sum()
        FP = (pred_mask * (1 - target_mask)).sum()
        FN = ((1 - pred_mask) * target_mask).sum()
        TN = ((1 - pred_mask) * (1 - target_mask)).sum()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
        iou = TP / (TP + FP + FN + 1e-8)

        class_name = class_names[cls] if class_names else f"Class_{cls}"
        metrics[class_name] = {
            "dice": dice.item(),
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "support": target_mask.sum().item(),
        }

    return metrics


def calculate_iou_score(preds, targets, num_classes, ignore_index=None, dilation=0):
    """
    Calculate IoU (Intersection over Union) for multi-class segmentation.
    More forgiving than Dice for boundary misalignments.

    IoU = Intersection / Union
    """
    iou_scores = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_mask = preds == cls
        target_mask = targets == cls

        if dilation > 0:
            struct = np.ones((dilation * 2 + 1, dilation * 2 + 1))

            pred_np = pred_mask.cpu().numpy()
            target_np = target_mask.cpu().numpy()

            if pred_np.ndim == 3:
                pred_dilated = np.stack(
                    [
                        binary_dilation(pred_np[i], structure=struct)
                        for i in range(pred_np.shape[0])
                    ]
                )
                target_dilated = np.stack(
                    [
                        binary_dilation(target_np[i], structure=struct)
                        for i in range(target_np.shape[0])
                    ]
                )
            else:  # 2D case
                pred_dilated = binary_dilation(pred_np, structure=struct)
                target_dilated = binary_dilation(target_np, structure=struct)

            pred_mask = torch.from_numpy(pred_dilated).to(pred_mask.device)
            target_mask = torch.from_numpy(target_dilated).to(target_mask.device)

        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()

        if union == 0:
            iou_scores.append(1.0)
        else:
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou.item())

    return np.mean(iou_scores)


def calculate_panoptic_quality(preds, targets, num_classes, iou_threshold=0.5):
    """
    Panoptic Quality (PQ) = Detection Quality x Segmentation Quality

    Used in CellVit, CoNIC challenge, etc.
    Matches nuclei instances first, then evaluates overlap.

    Args:
        preds: (B, H, W) predicted instance masks
        targets: (B, H, W) ground truth instance masks
        num_classes: number of classes
        iou_threshold: IoU threshold for matching (default 0.5)
    """
    pq_scores = []

    for cls in range(num_classes):
        pred_instances = torch.unique(preds[preds == cls])
        target_instances = torch.unique(targets[targets == cls])

        if len(pred_instances) == 0 and len(target_instances) == 0:
            pq_scores.append(1.0)
            continue

        iou_matrix = np.zeros((len(pred_instances), len(target_instances)))

        for i, pred_id in enumerate(pred_instances):
            pred_mask = preds == pred_id
            for j, target_id in enumerate(target_instances):
                target_mask = targets == target_id
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()
                iou_matrix[i, j] = (intersection / (union + 1e-8)).item()

        matched_pred, matched_target = linear_sum_assignment(-iou_matrix)

        TP = 0
        sum_iou = 0
        for p_idx, t_idx in zip(matched_pred, matched_target):
            if iou_matrix[p_idx, t_idx] >= iou_threshold:
                TP += 1
                sum_iou += iou_matrix[p_idx, t_idx]

        FP = len(pred_instances) - TP
        FN = len(target_instances) - TP

        if TP + FP + FN == 0:
            pq_scores.append(0.0)
        else:
            pq = sum_iou / (TP + 0.5 * FP + 0.5 * FN + 1e-8)
            pq_scores.append(pq)

    return np.mean(pq_scores)


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
        targets_one_hot = (
            F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )

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


class FocalTverskyLoss(nn.Module):
    """
    Multi-class Focal Tversky Loss from CellVit paper.
    Suitable for severe class imbalance.

    Tversky = (TP) / (TP + α FN + β FP)
    Focal = (1 - Tversky)^γ
    """

    def __init__(self, num_classes, alpha=0.7, beta=0.3, gamma=1.33, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        probs = F.softmax(logits, dim=1)

        # One-hot encode
        targets_1h = (
            F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )

        tversky_total = 0.0
        cls_count = 0

        for cls in range(self.num_classes):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue

            p = probs[:, cls, :, :]
            t = targets_1h[:, cls, :, :]

            TP = (p * t).sum(dim=(1, 2))
            FP = (p * (1 - t)).sum(dim=(1, 2))
            FN = ((1 - p) * t).sum(dim=(1, 2))

            tversky = (TP + 1e-6) / (TP + self.alpha * FN + self.beta * FP + 1e-6)
            focal_tversky = (1 - tversky) ** self.gamma

            tversky_total += focal_tversky.mean()
            cls_count += 1

        return tversky_total / cls_count


class CombinedLoss(nn.Module):
    """
    Combined Cross Entropy, Dice, and optional Focal Tversky Loss.

    total = α * CE + β * Dice + γ * FocalTversky
    """

    def __init__(
        self,
        num_classes,
        ce_weight=0.4,  # weight for CE
        dice_weight=0.4,  # weight for Dice
        ft_weight=0.2,  # weight for Focal Tversky
        ignore_index=None,
        class_weights=None,
        ft_alpha=0.7,  # Tversky α
        ft_beta=0.3,  # Tversky β
        ft_gamma=1.33,  # Focal exponent
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ft_weight = ft_weight

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )

        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

        self.focal_tversky = FocalTverskyLoss(
            num_classes=num_classes,
            alpha=ft_alpha,
            beta=ft_beta,
            gamma=ft_gamma,
            ignore_index=ignore_index,
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets.long()) * self.ce_weight
        dice_loss = self.dice(logits, targets.long()) * self.dice_weight
        ft_loss = self.focal_tversky(logits, targets.long()) * self.ft_weight

        return ce_loss + dice_loss + ft_loss
