import os
from pathlib import Path

# Detect environment (script vs. notebook)
if "__file__" in globals():
    # When running as a .py script
    ROOT_DIR = Path(__file__).resolve().parent
else:
    # When running inside a Jupyter notebook or interactive shell
    ROOT_DIR = Path().resolve()
    while not (ROOT_DIR / "src").exists() and ROOT_DIR != ROOT_DIR.parent:
        ROOT_DIR = ROOT_DIR.parent

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")

#Test data
TEST_DATA = os.path.join(DATA_DIR, "breast_tissue_crop.png")

