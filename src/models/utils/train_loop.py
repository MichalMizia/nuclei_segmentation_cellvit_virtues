import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.cellvit_decoder import CellViTDecoder
from src.utils.metrics import calculate_dice_score
from tqdm import tqdm


def compute_semantic_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Compute a 4-neighborhood semantic boundary map from an integer mask.

    Args:
        mask: (B, H, W) integer class map.

    Returns:
        (B, H, W) bool tensor where True indicates a boundary pixel.
    """
    if mask.dim() != 3:
        raise ValueError(f"Expected mask of shape (B,H,W), got {tuple(mask.shape)}")

    boundary = torch.zeros_like(mask, dtype=torch.bool)
    boundary[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    boundary[:, :-1, :] |= mask[:, :-1, :] != mask[:, 1:, :]
    boundary[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, :, :-1] |= mask[:, :, :-1] != mask[:, :, 1:]
    return boundary


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
    boundary_attention: bool = False,
    lambda_boundary: float = 0.05,
    use_feature_gating: bool = False,
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
            
            # Phase 3 – Step 1: Two-pass forward with feature gating
            if use_feature_gating:
                # First forward: Get initial predictions (no gating)
                decoder.use_feature_gating = False
                outputs = decoder(input)
                pred_logits = outputs["nuclei_type_map"]
                
                if pred_logits.shape[-2:] != mask.shape[-2:]:
                    pred_logits = torch.nn.functional.interpolate(
                        pred_logits,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                
                # Build foreground mask (background = 0)
                with torch.no_grad():
                    fg_mask = (pred_logits.argmax(1) != 0).float().unsqueeze(1)
                
                # Second forward: Apply gating with foreground mask
                decoder.use_feature_gating = True
                outputs = decoder(input, fg_mask=fg_mask)
                pred_logits = outputs["nuclei_type_map"]
                
                if pred_logits.shape[-2:] != mask.shape[-2:]:
                    pred_logits = torch.nn.functional.interpolate(
                        pred_logits,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                # Standard single-pass forward
                outputs = decoder(input)
                pred_logits = outputs["nuclei_type_map"]

                if pred_logits.shape[-2:] != mask.shape[-2:]:
                    pred_logits = torch.nn.functional.interpolate(
                        pred_logits,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            seg_loss = criterion(pred_logits, mask)
            if boundary_attention:
                if "boundary_map" not in outputs:
                    raise KeyError(
                        "boundary_attention=True but decoder did not return 'boundary_map'. "
                        "Instantiate CellViTDecoder(boundary_attention=True) or disable boundary_attention in train_loop."
                    )
                boundary_logits = outputs["boundary_map"]
                if boundary_logits.shape[-2:] != mask.shape[-2:]:
                    boundary_logits = torch.nn.functional.interpolate(
                        boundary_logits,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                boundary_target = compute_semantic_boundary(mask).unsqueeze(1).float()
                boundary_loss = F.binary_cross_entropy_with_logits(
                    boundary_logits, boundary_target
                )
                loss = seg_loss + lambda_boundary * boundary_loss
            else:
                loss = seg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if include_skip_connections:
                del he_img, intermediate_pss
            if boundary_attention:
                del boundary_logits, boundary_target, boundary_loss
            del mask, outputs, pred_logits, seg_loss, loss, input, pss

            torch.cuda.empty_cache()

        avg_train_loss = running_loss / steps if steps > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        decoder.eval()
        # Ensure gating is disabled during validation
        if use_feature_gating:
            decoder.use_feature_gating = False
        val_running_loss = 0.0
        val_running_dice = 0.0
        val_steps = 0

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
