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


class HVLoss(nn.Module):
    """
    HV Map Loss combining MSE and MSGE (Mean Squared Gradient Error).
    Used in CellVit for instance segmentation.

    L_HV = λ_MSE x MSE + λ_MSGE x MSGE

    Args:
        mse_weight (float): Weight for MSE loss (default: 1.0)
        msge_weight (float): Weight for gradient loss (default: 1.0)
    """

    def __init__(self, mse_weight=1.0, msge_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.msge_weight = msge_weight

    def forward(self, pred_hv, gt_hv):
        """
        Args:
            pred_hv (torch.Tensor): Predicted HV map (B, 2, H, W)
            gt_hv (torch.Tensor): Ground truth HV map (B, 2, H, W)
            instance_mask (torch.Tensor, optional): Instance mask (B, H, W), 0 = background

        Returns:
            torch.Tensor: Combined loss
            dict: Loss components for logging
        """
        mse_loss = F.mse_loss(pred_hv, gt_hv)

        # MSGE Loss (Mean Squared Gradient Error)
        # Horizontal gradients (B, 2, H, W-1)
        pred_grad_x = pred_hv[:, :, :, 1:] - pred_hv[:, :, :, :-1]
        gt_grad_x = gt_hv[:, :, :, 1:] - gt_hv[:, :, :, :-1]

        # Vertical gradients (B, 2, H-1, W)
        pred_grad_y = pred_hv[:, :, 1:, :] - pred_hv[:, :, :-1, :]
        gt_grad_y = gt_hv[:, :, 1:, :] - gt_hv[:, :, :-1, :]

        grad_loss_x = F.mse_loss(pred_grad_x, gt_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, gt_grad_y)

        msge_loss = grad_loss_x + grad_loss_y

        total_loss = self.mse_weight * mse_loss + self.msge_weight * msge_loss

        return total_loss


class InstanceSegLoss(nn.Module):
    """
    Combined loss for instance segmentation:
    - Binary segmentation loss: weighted sum of Focal Tversky and Dice loss
    - HV map loss: weighted sum of MSE and MSGE

    L_total = λ_bin_ft * FocalTversky + λ_bin_dice * Dice + λ_hv_mse * MSE + λ_hv_msge * MSGE

    Args:
        num_classes (int): Number of classes for binary segmentation (should be 2)
        bin_ft_weight (float): Weight for Focal Tversky loss (default: 1.0)
        bin_dice_weight (float): Weight for Dice loss (default: 1.0)
        hv_mse_weight (float): Weight for HV MSE loss (default: 1.0)
        hv_msge_weight (float): Weight for HV MSGE loss (default: 1.0)
        ft_alpha, ft_beta, ft_gamma: Focal Tversky parameters
    """

    def __init__(
        self,
        num_classes=2,
        ft_weight=1.0,
        dice_weight=1.0,
        hv_mse_weight=1.0,
        hv_msge_weight=1.0,
        ft_alpha=0.7,
        ft_beta=0.3,
        ft_gamma=1.33,
        ignore_index=None,
    ):
        super().__init__()
        self.ft_weight = ft_weight
        self.dice_weight = dice_weight
        self.hv_mse_weight = hv_mse_weight
        self.hv_msge_weight = hv_msge_weight

        self.focal_tversky = FocalTverskyLoss(
            num_classes=num_classes,
            alpha=ft_alpha,
            beta=ft_beta,
            gamma=ft_gamma,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.hv_loss = HVLoss(mse_weight=hv_mse_weight, msge_weight=hv_msge_weight)

    def forward(self, binary_logits, binary_targets, hv_pred, hv_gt):
        """
        Args:
            binary_logits (torch.Tensor): (B, 2, H, W) - logits for binary segmentation
            binary_targets (torch.Tensor): (B, H, W) - ground truth mask (0/1)
            hv_pred (torch.Tensor): (B, 2, H, W) - predicted HV map
            hv_gt (torch.Tensor): (B, 2, H, W) - ground truth HV map
        Returns:
            torch.Tensor: total loss
        """
        bin_ft_loss = self.focal_tversky(binary_logits, binary_targets) * self.ft_weight
        bin_dice_loss = self.dice(binary_logits, binary_targets) * self.dice_weight

        hv_loss = self.hv_loss(hv_pred, hv_gt)  # already weighted inside HVLoss

        total_loss = bin_ft_loss + bin_dice_loss + hv_loss
        return total_loss


def print_class_statistics(loader, model, device, num_classes):
    with torch.no_grad():
        pred_class_counts = torch.zeros(num_classes, dtype=torch.long)
        gt_class_counts = torch.zeros(num_classes, dtype=torch.long)

        all_preds = []
        all_masks = []

        for batch in loader:
            pss, mask, he_img, intermediate_pss = batch

            pss = pss.to(device)
            mask = mask.to(device).long()

            he_img = he_img.to(device)
            intermediate_pss = [ip.to(device) for ip in intermediate_pss]
            input = [he_img] + intermediate_pss + [pss]

            outputs = model(input)
            pred_logits = outputs["nuclei_type_map"]

            # class counts
            pred_mask = torch.argmax(pred_logits, dim=1)  # (B, H, W)
            for c in range(num_classes):
                pred_class_counts[c] += (pred_mask == c).sum().item()
                gt_class_counts[c] += (mask == c).sum().item()

            all_preds.append(pred_mask.cpu())
            all_masks.append(mask.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        stats = calculate_per_class_metrics(all_preds, all_masks, num_classes)

        pred_class_counts = pred_class_counts * 100 / pred_class_counts.sum()
        gt_class_counts = gt_class_counts * 100 / gt_class_counts.sum()
        for c in range(num_classes):
            print(
                f"Class {c}: Pred = {pred_class_counts[c].item():.2f}%, Ground Truth = {gt_class_counts[c].item():.2f}%, Dice = {stats[f'Class_{c}']['dice']:.4f}, Recall = {stats[f'Class_{c}']['recall']:.4f}"
            )
