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

#Orion Subset
ORION_SUBSET_DIR = "/data/virtues_orion_dataset/virtues_example/orion_subset"

#Test data
TEST_DATA = os.path.join(DATA_DIR, "breast_tissue_crop.png")

#Resources
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
IMAGE_DIR = os.path.join(RESOURCES_DIR, "images")


#Virtues weights
VIRTUES_WEIGHTS_PATH = "/data/virtues_orion_dataset/virtues_example/virtues_weights"