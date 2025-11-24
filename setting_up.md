# CellViT Inference on EPFL RCP Setum

---

## 1. Connect to the pod and clone the repository 

```bash
kubectl exec -it -n runai-course-cs-433-group01-kuci tissuevit-full-0-0 -- /bin/bash

cd /data
mkdir -p code
cd code
```

Clone the repo if not already there.
```bash
git clone https://<Personal Github Token>@github.com/CS-433/project-2-gradient_tri_scent.git
git config --global --add safe.directory /data/code/project-2-gradient_tri_scent

cd project-2-gradient_tri_scent
```
The token must have repository read and write permission.
Replace `<Personal Github Token>` with your actual GitHub personal access token.

---

## 2. Run CellViT

Place your input images / WSIs inside the data/ folder of this repository, for example:

```bash
/data/code/project-2-gradient_tri_scent/data/breast_tissue_crop.png
```
The train.py script expects the test image path to live under data/ (configured via config.py).

---

## 2. Activate the tissuevit environment

```bash
source /opt/conda/bin/activate tissuevit
```

The `tissuevit` environment includes:

- PyTorch with GPU support  
- `cellvit` (CellViT-Inference)  
- `openslide-python` and `openslide-bin`  
- `pyvips` and `libvips` (for WSI-compatible TIFFs)

If the cellvit is not there install with pip install cellvit.

---


## 3. Run CellViT

The `train.py` script:

- Takes an input slide (or WSI-compatible TIFF)  
- Invokes `cellvit-inference` with:  
  - `--model SAM`  
  - `--nuclei_taxonomy pannuke`  
  - `--gpu 0`  
- Runs **CellViT-SAM-H** on the slide  
- Writes outputs to: `/data/cellvit_out`

Run it:

```bash
python train.py
```

Run any other script:

```bash
python the_file_that_you_want_to.py
```

---

## 4. Outputs

CellViT results (tables, logs, GeoJSONs if enabled) are stored in:

```bash
ls /data/cellvit_out
```

Inspect using `ls`, `head`, or a small Python script.

