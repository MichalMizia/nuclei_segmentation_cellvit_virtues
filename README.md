# Cell Instance and Semantic Segmentation using H&E and SP Data - CS-433 Machine Learning Project 2

**Team Members:** Michal Mizia, Jon Kuci, Huaqing Li

## Project Overview

This project investigates advanced methods for **semantic and instance segmentation of nuclei** in tissue images, comparing the performance of different encoder architectures and imaging modalities. We leverage two main types of input data:

- **H&E images:** traditional histopathology staining providing strong structural cues for nuclei boundaries.
- **Spatial Proteomics (SP) images:** multiplexed protein expression at single-cell resolution, highlighting class-discriminative features.

The goal is to evaluate how these modalities affect both **semantic segmentation** and the generation of **pixel-wise representations** needed for instance segmentation.

Key highlights:

- Comparison of the **CellViT encoder** (trained on H&E) and the **Virtues encoder** (trained on SP data).
- Evaluation of semantic segmentation performance via **Dice scores** and class-wise predictions.
- Assessment of instance segmentation quality using **binary nuclei maps** and **horizontal-vertical (HV) distance maps**, following the HoVer-Net paradigm.
- Investigation of how H&E and SP inputs influence the **quality of HV and binary map predictions**, which are critical for separating touching or overlapping nuclei.

The project demonstrates that **Virtues encoders** provide more semantically meaningful features, while H&E images slightly outperform SP data for precise instance segmentation maps due to their structural clarity. Both modalities, however, produce high-quality instance segmentation maps when combined with the appropriate decoder architecture and loss functions.

**Important:** only during the last day of the project we learned from the lab that when passing H&E images through the VirTues encoder, even tough they get passed by all the transformer layers, the encoder was not trained on them so th transformer weights for H&E images are random, and there should be no meaningful difference between passing SP only or SP+H&E modality to the VirTues encoder. This was impossible to know beforehand, and you can probably imagine that this experience was also frustrating for us. In the end, we didn't have time to remove the SP+H&E experiments from notebook Part2_cellvit_virtues_model.ipynb. We hope this mistake gets excused.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [Experiments](#experiments)
- [Results](#results)
- [Report](#report)

---

## Project Structure

```
project-2-gradient_tri_scent/
├── README.md                   # This file
├── config.py                   # File with project configs such as paths
│
├── src/          # Main package (extended implementation)
│   ├── models/
|   |   ├── wrappers/
|   |   |   ├── cellvit_wrapper.py    # used to produce precomputed embeddings for the whole DS with CellVit encoder
|   |   |   └──virtues_wrapper.py    # used to produce precomputed embeddings for the whole DS with VirTues encoder
|   |   ├── utils/
|   |   |   └──  # model utilities, to compute class/patch weights, basic model blocks, train loop for decoder
│   │   ├── cellvit_decoder.py        # adapted cellvit decoder model for semantic multi-class segmentation
│   │   └── instance_mask_decoder.py  # adapted cellvit decoder model for instance single-class segmentation
│   │
│   ├── dataset/datasets/ # mostly classes provided to us by the lab except:
│   │   ├── embeddings_dataset.py       # 2 pytorch datasets for storing base images, masks etc, with or without precomputed embeddings
│   │   └── instance_mask_dataset.py    # same as above with addition of precomputed hv and instance masks
│   │
│   ├── modules/ # virtues encoder provided to us by the lab
│   │   └── flex_dual_virtues_new_init.py    # modified by us to return intermediate representations
│   │
│   └── utils/ # utility files
│         └── metrics.py    # result metrics and all loss functions
│
├── notebooks/          # Interactive notebooks
│   ├── Part1_cellvit_inference.ipynb            # Training cellvit decoder for pretrained cellvit encoder
│   ├── Part2_cellvit_virtues_model.ipynb        # Training cellvit decoder for virtues encoder, different versions
│   ├──  Part3_Instance_Segmentation.ipynb        # Training hv and binary map decoder for both virtues and cellvit encoders
│   └── Part4_cellvit_architectural_variants.ipynb    # Decoder-level architectural extensions to the CellViT
│
├── experiments/ # non-interactive experiment scripts (EPFL RCP)
│   ├── Part4_*.py           # decoder-level architectural variants (Part 4)
│   ├── *.py                 # other experiment files (Parts 1–3)
│   ├── *_results.pkl        # serialized experiment results
│   └── Experiments.ipynb    # visualization and comparison of results
│
└── scripts # scripts for running the interactive pod and the non-interactive experiments

```

---

## Quick Start

### 1. Environment Setup

If you have not used RCP before, go to https://wiki.rcp.epfl.ch/home/CaaS/Onboarding_Fast_Track and make sure you have runai and kubernetes installed. Also, either be on EPFL Wi-Fi or connect to the EPFL VPN via the Cisco Client.

To monitor your runai job, open https://rcpepfl.run.ai/ and login with SSO.

### 2. Running the jobs

- To run the interactive job for viewing the dataset and project, open a bash terminal and run `./scripts/run_tissuevit.sh`.
- To run the experiment, in `./scripts/run_experiment.sh` replace the file paths with your paths, make sure you have the python files already on the pod in the correct spot (either on scratch or home storage), and run the file.

**Useful kubernetes commands:**

```sh
USER="your-gaspar"
NAMESPACE="runai-course-cs-433-group01-${USER}"

# Get available pods
kubectl get pods -n $NAMESPACE

# Open a bash shell on the main pod (replace 'tissuevit-0-0' with your pod name if different)
kubectl exec -it -n $NAMESPACE tissuevit-0-0 -- /bin/bash

# Get logs from the main pod
kubectl logs -n $NAMESPACE tissuevit-0-0

# Get logs from the experiment pod and follow output (replace 'tissuevit-experiment-0-0' with your pod name if different)
kubectl logs -n $NAMESPACE tissuevit-experiment-0-0 --follow

# Describe pod events for debugging
kubectl describe pod -n $NAMESPACE tissuevit-0-0
```

**Once connected to the pod, run:**

```sh
cd /
# Activate the environment
source /opt/conda/bin/activate tissuevit
# paste these lines to avoid incompatibility errors when using virtues down the line
pip uninstall -y flash-attn torch torchvision torchaudio xformers && \
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --extra-index-url https://download.pytorch.org/whl/cu124 && \
pip install flash-attn --no-build-isolation --no-cache-dir && \
conda install -c conda-forge openslide libvips pyvips -y && \
pip install kornia cellvit && \
pip install --upgrade setuptools
```

**To run an interactive notebook**:

```sh
# Register the kernel for jupyter.
python -m ipykernel install --user --name tissuevit --display-name "tissuevit"
# run jupyter on pod
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Then port-forward in a terminal on your local machine:** `kubectl port-forward -n runai-course-cs-433-group01-mizia tissuevit-0-0 8888:8888`

---

## Notebooks

### Part1_cellvit_inference.ipynb

This notebook demonstrates training a semantic segmentation decoder for the pretrained CellViT encoder using H&E images. It begins by exploring key dataset properties, such as cell counts, cell types, and tissue IDs, then visualizes input H&E images. The decoder is trained on the semantic segmentation task, and predicted segmentation maps are visualized to assess performance qualitatively. Predicted dice scores for each cell class are also assesed.

### Part2_cellvit_virtues_model.ipynb

This notebook trains semantic segmentation decoders for the Virtues encoder using two input configurations: SP-only and SP + H&E data. It visualizes input images, trains the decoders on the semantic segmentation task, and compares predicted segmentation maps to assess the effect of input modalities on performance.

### Part3_Instance_Segmentation.ipynb

This notebook evaluates the instance segmentation capabilities of both CellViT and Virtues encoders using horizontal–vertical (HV) distance maps and binary nuclei maps. It trains the same decoder for each encoder to predict HV and binary maps, visualizes ground-truth and predicted outputs for comparison, and computes metrics such as Dice score and HV MSE.

### Part4_cellvit_architectural_variants.ipynb

This notebook investigates decoder-level architectural extensions to the CellViT framework when operating on Spatial Proteomics (SP) embeddings produced by a frozen VirTues encoder. All experiments share the same encoder, data split, and training protocol, and differ only in the decoder architecture.

The following decoder variants are evaluated:

- Baseline SP-only decoder
- Boundary supervision, using an auxiliary boundary prediction head
- Masked self-attention, inspired by Masked2Former, applied at the final decoder stage
- Global context modeling, using self-attention at the decoder bottleneck

The experiments analyze convergence behavior, peak Dice performance, and stability across training epochs, highlighting boundary supervision as the most reliable improvement and masked attention as a higher-capacity but more expensive refinement.


## Experiments


### Data Splitting and Training Protocol

For the first three experiments, each Whole Slide Image (WSI) dataset was partitioned into smaller subsets to accelerate training. Two folds were created:

- **Fold 1:** train = first 25% of items, test = next 25%.
- **Fold 2:** train = third 25% of items, test = last 25%.

Models were trained for 30 epochs using AdamW optimizer with learning rate 5e-4, weight decay 1e-2, and a cosine annealing scheduler down to 1e-6.

The augmentation experiment used full 3-fold cross-validation and 50 epochs because we expected it might take longer for the model to converge with perturbations in the original image.

### criterion_experiment.py

This experiment evaluates the effect of different segmentation loss formulations on nuclei segmentation performance. Total loss combines CE and Dice losses, optionally with Focal Tversky loss to emphasize underrepresented classes.

### skipconn_experiment.py

We test the impact of incorporating skip connections from multiple encoder depths, as well as the original input image, compared to using only the final encoder embeddings. This helps understand the importance of early-layer features for reconstructing fine-grained spatial details.

### oversampling_experiment.py

A patch-level oversampling strategy is applied to address class imbalance in nuclei types. Each training patch is weighted based on the presence of rare cell classes, allowing the model to better learn underrepresented types.

### augmentation_experiment.py

A lightweight augmentation pipeline is tested to improve generalization. This includes geometric transformations, H&E color jitter, and CyCIF channel dropout to increase robustness to signal variability.

### Hyperparameters

For all notebooks, the encoders were kept frozen and only the decoder parameters were optimized. Precomputed embeddings were used with a tissue-level sequential split (80% train, 20% validation), sorted alphabetically by tissue identifier.

Training employed the AdamW optimizer with a learning rate of 5×10⁻⁴ and weight decay of 1×10⁻², combined with a cosine annealing warm restart scheduler (`T_0 = 20`, `η_min = 10⁻⁶`). Models were trained for up to 100 epochs with early stopping patience of 30 and a batch size of 128. Each 3000×3000 pixel SP image was split into 625 smaller images of size 120×120 pixels to fit into GPU memory. A single A100-SXM4-80GB Nvidia GPU was used for training.

For the first three experiments, each Whole Slide Image (WSI) dataset was partitioned into smaller subsets to accelerate training:

- **Fold 1:** train set = first 25% of all items, test set = next 25%.
- **Fold 2:** train set = third 25% of all items, test set = last 25%.

Each experiment was run using only the corresponding fold's training and test sets. Models were trained for 30 epochs using the AdamW optimizer with learning rate 5×10⁻⁴ and weight decay 1×10⁻², together with a cosine annealing learning rate scheduler with a minimum learning rate of 10⁻⁶. Only in the augmentation experiment did we aim for full 3-fold cross-validation, as overly small train sets might not work well with data augmentation, and for 50 epochs as perturbations to training images cause the model to converge more slowly.


### Part4: CellViT Architectural Variants (Decoder-Level)


For Part 4, we evaluate decoder-level architectural variants using a frozen VirTues encoder and SP-only precomputed embeddings. A tissue-level sequential split is used, with 80% of tissues for training and 20% for validation. Models are trained using the AdamW optimizer with a learning rate of 5e-4 and weight decay of 1e-2, together with a cosine annealing warm restart scheduler (T₀ = 20, η_min = 1e-6). Training is run for up to 86 epochs with early stopping patience set to 30 epochs and a batch size of 128. All experiments are executed on a single A100-SXM4-80GB GPU. All Part 4 experiments use identical training settings and differ only in the decoder architecture.

Each decoder configuration is trained independently on a single fixed split. For every run, training loss, validation loss, and validation Dice scores are recorded at each epoch and saved to disk as `.npz` files. The training scripts for Part 4 are located under the `experiments/` directory:

- `experiments/Part4_sp_only.py`
- `experiments/Part4_sp_boundary_att.py`
- `experiments/Part4_masked_att.py`
- `experiments/Part4_global_context.py`

Final results and convergence behavior are analyzed and visualized in the corresponding results notebook.


---

## Report

A comprehensive 4-page LaTeX report is available in our [latest release](https://github.com/CS-433/project-2-gradient_tri_scent/releases/latest)

The report covers:

## Contact

For questions about this implementation:

- Michal MIZIA
- Jon Kuci
- Huaqing Li

**GitHub Repository:** https://github.com/CS-433/project-2-gradient_tri_scent
