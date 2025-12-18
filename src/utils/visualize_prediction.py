import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt




def visualize_single_decoder_prediction(
    decoder,
    vis_ds,
    ds,
    device,
    num_classes: int,
    calculate_dice_score,
    idx: int = -1,
    cmap: str = "tab20",
    figsize=(12, 5),
):
    """
    Visualize a single prediction from a decoder.

    Parameters
    ----------
    decoder : torch.nn.Module
        Trained decoder model.
    vis_ds : Dataset
        EmbeddingsDataset.
    ds : Dataset
        Original dataset (used to retrieve full H&E image).
    device : torch.device
        CUDA or CPU device.
    num_classes : int
        Number of nuclei classes.
    calculate_dice_score : callable
        Dice computation function.
    idx : int, optional
        Index in vis_ds to visualize. Default -1 (last sample).
    cmap : str, optional
        Colormap for segmentation masks.
    figsize : tuple, optional
        Matplotlib figure size.
    """

    decoder.eval()

    if idx < 0:
        idx = len(vis_ds) + idx  # allow -1, -2, ...

    with torch.no_grad():
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        tid, img, mask, pss, intermediate_pss = vis_ds[idx]

        img = img.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device).long()
        pss = pss.unsqueeze(0).to(device)
        intermediate_pss = [
            ip.unsqueeze(0).to(device) for ip in intermediate_pss
        ]

        decoder_input = [img] + intermediate_pss + [pss]

        outputs = decoder(decoder_input)
        pred_logits = outputs["nuclei_type_map"]
        pred_mask = torch.argmax(pred_logits, dim=1)

        dice = calculate_dice_score(pred_mask, mask, num_classes)

        # Load full H&E image
        he_img = ds[0].unimodal_datasets["he"]._get_tissue_all_channels(tid)
        he_vis = (
            he_img.transpose(1, 2, 0)
            if he_img.shape[0] == 3
            else he_img[0]
        )

        # Plot
        axes[0].imshow(he_vis)
        axes[0].set_title("H&E")
        axes[0].axis("off")

        axes[1].imshow(
            mask[0].cpu().numpy(),
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(
            pred_mask[0].cpu().numpy(),
            cmap=cmap,
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[2].set_title(f"Prediction (Dice: {dice:.3f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        # Cleanup
        del (
            img,
            mask,
            pss,
            intermediate_pss,
            decoder_input,
            outputs,
            pred_logits,
            pred_mask,
        )
        torch.cuda.empty_cache()