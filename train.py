import torch
import numpy as np
from pathlib import Path
from PIL import Image

from cellvit.inference.inference import CellViTInference
from config import TEST_DATA


def to_numpy(x):
    """Convert torch or numpy to plain numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    # 1. Config
    model_name = "cellvit_sam_h_pannuke"   # your PanNuke model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = 0 if device == "cuda" else -1

    img_path = Path(TEST_DATA)
    if not img_path.exists():
        raise FileNotFoundError(f"TEST_DATA does not exist: {img_path}")

    print(f"Using image: {img_path}")
    print(f"Device: {device} (gpu={gpu_id})")
    print(f"Model: {model_name}")

    # 2. Init inference engine (this will download weights if needed)
    inference_runner = CellViTInference(
        model_name=model_name
    )

    # 3. Run inference on the PNG
    #    run_inference should handle patching / tiling internally.
    results = inference_runner.run_inference(str(img_path))

    # Expected keys: "instance_map", "type_map"
    instance_map = to_numpy(results["instance_map"])
    type_map = to_numpy(results["type_map"])

    # 4. Summary in console
    num_cells = int(instance_map.max())
    print(f"Inference complete. Found {num_cells} cells.")
    print(f"instance_map shape: {instance_map.shape}, dtype: {instance_map.dtype}")
    print(f"type_map shape: {type_map.shape}, dtype: {type_map.dtype}")

    # 5. Save outputs as images next to the script
    # instance_map can exceed 255, so store as 16-bit PNG
    inst_out = Path("instance_map.png")
    type_out = Path("type_map.png")

    inst_img = Image.fromarray(instance_map.astype(np.uint16))
    inst_img.save(inst_out)

    type_img = Image.fromarray(type_map.astype(np.uint8))
    type_img.save(type_out)

    print(f"Saved instance map to {inst_out.resolve()}")
    print(f"Saved type map to {type_out.resolve()}")


if __name__ == "__main__":
    main()
