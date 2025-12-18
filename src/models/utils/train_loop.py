import torch
from torch.utils.data import DataLoader
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import calculate_dice_score
from tqdm import tqdm


def train_loop(
    train_loader: DataLoader,
    test_loader: DataLoader,
    decoder: CellViTDecoder,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    num_classes: int = 10,
    device: str = "cuda",
    save_path: str | None = "./model.pth",
    include_skip_connections: bool = False,
    use_tqdm: bool = False,
    verbose: bool = True,
):
    """
    Used to train the decoder model.
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_dices: List of validation Dice scores per epoch
    """
    train_losses = []
    val_losses = []
    val_dices = []
    best_val_dice = 0.0
    early_stop_epochs = 0

    for epoch in range(num_epochs):
        # --- Training Phase ---
        decoder.train()
        running_loss = 0.0
        steps = 0

        if use_tqdm:
            train_iterator = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
            )
        else:
            train_iterator = train_loader
        for batch in train_iterator:
            pss, mask, he_img, intermediate_pss = batch

            pss = pss.to(device)
            mask = mask.to(device).long()

            if include_skip_connections:
                he_img = he_img.to(device)
                intermediate_pss = [ip.to(device) for ip in intermediate_pss]
                input = [he_img] + intermediate_pss + [pss]
            else:
                input = pss

            optimizer.zero_grad()
            outputs = decoder(input)
            pred_logits = outputs["nuclei_type_map"]

            if pred_logits.shape[-2:] != mask.shape[-2:]:
                pred_logits = torch.nn.functional.interpolate(
                    pred_logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(pred_logits, mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if include_skip_connections:
                del he_img, intermediate_pss
            del mask, outputs, pred_logits, loss, input, pss

            torch.cuda.empty_cache()

        avg_train_loss = running_loss / steps if steps > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        decoder.eval()
        val_running_loss = 0.0
        val_running_dice = 0.0
        val_steps = 0

        pred_class_counts = torch.zeros(num_classes, dtype=torch.long)
        gt_class_counts = torch.zeros(num_classes, dtype=torch.long)
        max_pred_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                pss, mask, he_img, intermediate_pss = batch
                pss = pss.to(device)
                mask = mask.to(device).long()

                if include_skip_connections:
                    he_img = he_img.to(device)
                    intermediate_pss = [ip.to(device) for ip in intermediate_pss]
                    input = [he_img] + intermediate_pss + [pss]
                else:
                    input = pss

                outputs = decoder(input)
                pred_logits = outputs["nuclei_type_map"]

                if pred_logits.shape[-2:] != mask.shape[-2:]:
                    pred_logits = torch.nn.functional.interpolate(
                        pred_logits,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss = criterion(pred_logits, mask)
                val_running_loss += loss.item()

                pred_mask = torch.argmax(pred_logits, dim=1)
                dice = calculate_dice_score(pred_mask, mask, num_classes)
                val_running_dice += dice

                # DEBUG: Collect statistics
                for c in range(num_classes):
                    pred_class_counts[c] += (pred_mask == c).sum().cpu()
                    gt_class_counts[c] += (mask == c).sum().cpu()
                
                # Track max probabilities (confidence)
                probs = torch.softmax(pred_logits, dim=1)
                max_prob, _ = probs.max(dim=1)
                max_pred_probs.append(max_prob.mean().cpu().item())

                val_steps += 1
                del pss, mask, outputs, pred_logits, loss, pred_mask

        avg_val_loss = val_running_loss / val_steps if val_steps > 0 else 0
        avg_val_dice = val_running_dice / val_steps if val_steps > 0 else 0

        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        # --- Scheduler Step ---
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}"
            )
            
            # DEBUG: Print prediction distribution
            total_preds = pred_class_counts.sum().item()
            total_gt = gt_class_counts.sum().item()
            avg_confidence = sum(max_pred_probs) / len(max_pred_probs) if max_pred_probs else 0
            
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Prediction Distribution:")
            for c in range(num_classes):
                pred_pct = 100 * pred_class_counts[c].item() / total_preds if total_preds > 0 else 0
                gt_pct = 100 * gt_class_counts[c].item() / total_gt if total_gt > 0 else 0
                print(f"    Class {c}: Pred={pred_pct:5.2f}% | GT={gt_pct:5.2f}%")
            
            # Check for collapse (>50% predictions in one class)
            dominant_class = pred_class_counts.argmax().item()
            dominant_pct = 100 * pred_class_counts[dominant_class].item() / total_preds if total_preds > 0 else 0
            if dominant_pct > 50:
                print(f"\033[91m  ⚠️  WARNING: Model collapsing to class {dominant_class} ({dominant_pct:.1f}%)\033[0m")

        # --- Save Best Model (Based on Dice) ---
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            early_stop_epochs = 0
            if verbose:
                print(
                    f"\033[46m  -->  New Best Dice: {best_val_dice:.4f} (Saved to {save_path})\033[0m"
                )
            if save_path is not None:
                torch.save(decoder.state_dict(), save_path)
        else:
            early_stop_epochs += 1
            if early_stop_epochs >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stop_epochs} epochs without improvement."
                )
                break

    return decoder, train_losses, val_losses, val_dices
