import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize_prediction(image_tensor, model_output, channel_idx=0):
    """
    Args:
        image_tensor: Input image [C, H, W] or [1, C, H, W]
        model_output: Dictionary from the model forward pass
        channel_idx: Which channel of the input to show (0 is usually DAPI/Nuclei)
    """
    # 1. Prepare Input Image for Display
    # Take first item in batch, select specific channel (DAPI), move to CPU
    if image_tensor.dim() == 4:
        img_display = image_tensor[0, channel_idx, :, :].cpu().numpy()
    else:
        img_display = image_tensor[channel_idx, :, :].cpu().numpy()

    # 2. Prepare Binary Mask (Cell Detection)
    # output["nuclei_binary_map"] shape: [1, 2, H, W]
    # We take argmax to get 0 (background) or 1 (cell)
    binary_logits = model_output["nuclei_binary_map"]
    binary_mask = torch.argmax(binary_logits, dim=1)[0].cpu().numpy()

    # 3. Prepare Cell Type Map (Classification)
    # output["nuclei_type_map"] shape: [1, NumClasses, H, W]
    type_logits = model_output["nuclei_type_map"]
    type_mask = torch.argmax(type_logits, dim=1)[0].cpu().numpy()

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Input (DAPI)
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title(f"Input Image (Channel {channel_idx})")
    axes[0].axis('off')

    # Plot Binary Prediction
    axes[1].imshow(binary_mask, cmap='jet', interpolation='nearest')
    axes[1].set_title("Predicted Binary Mask (Cell vs BG)")
    axes[1].axis('off')

    # Plot Type Prediction
    # We use a distinct colormap to see different cell types
    axes[2].imshow(type_mask, cmap='tab10', interpolation='nearest')
    axes[2].set_title("Predicted Cell Types")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# --- RUN THE VISUALIZATION ---
# We assume 'mx_images' is your input list and 'output' is your model result
# We wrap mx_images in torch.stack if it's a list, to get [B, C, H, W]
if isinstance(mx_images, list):
    input_tensor = torch.stack(mx_images)
else:
    input_tensor = mx_images

visualize_prediction(input_tensor, output, channel_idx=0)