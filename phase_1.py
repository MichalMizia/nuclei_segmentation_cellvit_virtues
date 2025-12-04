import subprocess
from pathlib import Path
import os

from config import TEST_DATA  # breast_tissue_crop.png lives here


def ensure_pyramidal_tiff(png_path: Path) -> Path:
    """
    Convert a PNG (or other flat image) to a pyramidal TIFF
    that CellViT can use via OpenSlide.

    Uses pyvips (no sudo needed).
    """
    import pyvips  # requires: pip install pyvips + conda install -c conda-forge libvips

    png_path = png_path.resolve()
    #  this is the correct way: change the *name*, not the suffix
    tiff_path = png_path.with_name(png_path.stem + "_pyramidal.tiff")

    if tiff_path.exists():
        print(f"[INFO] Pyramidal TIFF already exists: {tiff_path}")
        return tiff_path

    print(f"[INFO] Converting {png_path} -> {tiff_path} (pyramidal TIFF via pyvips)")

    img = pyvips.Image.new_from_file(str(png_path), access="sequential")
    img.tiffsave(
        str(tiff_path),
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        compression="jpeg",
        Q=90,
    )

    print(f"[INFO] Saved pyramidal TIFF: {tiff_path}")
    return tiff_path

def run_cellvit_on_wsi(wsi_path: Path):
    """
    Call cellvit-inference CLI from Python on a single WSI.
    Uses GPU 0, SAM model, PanNuke taxonomy.
    """
    wsi_path = wsi_path.resolve()
    outdir = Path("/data/cellvit_out").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cellvit-inference",
        "--model", "SAM",                 # SAM-based CellViT
        "--nuclei_taxonomy", "pannuke",   # PanNuke classes
        "--gpu", "0",                     # GPU ID (set -1 for CPU)
        "--enforce_amp",                  # mixed precision
        "--batch_size", "4",
        "--outdir", str(outdir),
        # processing mode: single WSI
        "process_wsi",
        "--wsi_path", str(wsi_path),
        "--wsi_mpp", "0.25",              # approx 40x
        "--wsi_magnification", "40",
    ]

    print("[INFO] Running CellViT with command:")
    print(" ".join(cmd))

    env = os.environ.copy()
    env["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

    subprocess.run(cmd, check=True, env=env)

    print(f"[INFO] CellViT finished. Outputs in: {outdir}")


def main():
    png_path = Path(TEST_DATA)
    if not png_path.exists():
        raise FileNotFoundError(f"TEST_DATA does not exist: {png_path}")

    print(f"[INFO] Using input image: {png_path}")

    # 1) Convert PNG to pyramidal TIFF 
    wsi_path = ensure_pyramidal_tiff(png_path)

    # 2) Run CellViT inference via CLI
    run_cellvit_on_wsi(wsi_path)


if __name__ == "__main__":
    main()