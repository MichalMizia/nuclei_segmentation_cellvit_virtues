# Phase 2 Report: VirTues + CellViT Segmentation Analysis

## 1. Executive Summary
**Status:** The current implementation in `cellvit_segmentation.ipynb` is functionally complete (it runs) but architecturally flawed for the segmentation task.
**Key Findings:**
1.  **Structural Incompatibility:** The `VirTues` encoder (by design) does not return the multi-scale feature maps required by the `CellViT` decoder.
2.  **Data Filtering:** The dataset loader (`MultiplexDataset`) actively filters out channels without protein embeddings. **DAPI (DNA stain)** is likely being removed because it is not a protein, depriving the model of the most critical signal for nuclei segmentation.
3.  **Resolution Mismatch:** The model is trying to segment fine nuclear boundaries from a coarse `16x16` feature map without access to high-resolution spatial information.

---

## 2. Deep Dive: Architecture Analysis

### The VirTues Encoder (`src/modules/flex_dual_virtues`)
I analyzed `FlexDualVirTuesEncoder` in `flex_dual_virtues_new_init.py`.
*   **Mechanism:** It processes a list of channel tensors (`multiplex`) and H&E images (`he`).
*   **Output:** The `forward_list` method returns:
    *   `x`: Channel-specific tokens ($C \times H \times W \times D$).
    *   `ps`: Patch Summary tokens ($H \times W \times D$).
*   **Limitation:** It **does not** return intermediate feature maps from different depths of the transformer (e.g., layers 3, 6, 9, 12).
*   **Impact on CellViT:** The `CellViT` decoder expects a U-Net-like structure with skip connections (`z0`, `z1`, `z2`, `z3`, `z4`) representing features at different scales (high-res to low-res).
    *   **Your Implementation:** You are forcing the single, low-resolution output `ps` (reshaped `z_final`) into *every* stage of the decoder.
    *   **Result:** The decoder has no high-frequency information to reconstruct sharp boundaries. It is effectively "hallucinating" shapes from a 16x16 grid.

### The Dataset Loader (`src/dataset/datasets/multiplex_base.py`)
I analyzed `MultiplexDataset`.
*   **Channel Filtering Logic:**
    ```python
    if uniprot_id in self.uniprot_to_index:
        mask_channels_with_embeddings.append(True)
    else:
        mask_channels_with_embeddings.append(False)
    ```
    Then later:
    ```python
    img = img[bmask_channels_with_embeddings]
    ```
*   **The DAPI Problem:** DAPI is a DNA stain, not a protein. It likely does not have a UniProt ID or a pre-computed protein embedding in `esm2_t30_150M_UR50D`. Therefore, **it is being silently discarded** by the loader.
*   **Consequence:** You are asking the model to segment nuclei without seeing the nuclei (DAPI), relying only on cytoplasmic protein signals which are often absent in the nucleus (negative stain).

---

## 3. Data Flow Diagram (Revised)

```mermaid
graph TD
    subgraph "Data Loading (src/dataset)"
        A[Raw Images] --> B[MultiplexDataset]
        B -->|Filter| C{Has Protein Embedding?}
        C -- Yes --> D[Keep Channel]
        C -- No (DAPI) --> E[Discard Channel]
        D --> F[Input Tensor: B, C_prot, 128, 128]
    end

    subgraph "VirTues Encoder (src/modules)"
        F --> G[FlexDualVirTuesEncoder]
        G -->|Forward| H[Transformer Layers]
        H -->|Final Layer Only| I[PSS Tokens: B, 16, 16, 512]
    end

    subgraph "CellViT Decoder (Your Implementation)"
        I --> J[Reshape to B, 512, 16, 16]
        J --> K[z4 (Bottleneck)]
        J --> L[z3 (Skip 3)]
        J --> M[z2 (Skip 2)]
        J --> N[z1 (Skip 1)]
        J --> O[z0 (Input Skip)]
        
        Note right of O: CRITICAL FAIL: z0 should be 128x128 image,\nbut is 16x16 feature map.
        
        K & L & M & N & O --> P[Decoder Upsampling]
        P --> Q[Segmentation Map]
    end
```

---

## 4. Recommendations for Phase 3

To make this work, you need to address the two main blockers: **Signal (DAPI)** and **Resolution (Skip Connections)**.

### A. Fix the Data (Crucial)
You must modify `MultiplexDataset` (or subclass it) to allow DAPI to pass through.
1.  **Identify DAPI:** Check the `channels.csv` or `channels_per_image.csv` to find the channel name for DAPI.
2.  **Bypass Filter:** Modify the loading logic to keep DAPI even if it has no embedding.
3.  **Embedding Strategy:** Since VirTues expects an embedding for every channel:
    *   **Option 1:** Assign a learnable embedding vector specifically for DAPI.
    *   **Option 2:** Use a "null" or "zero" embedding for it, but let the pixel values pass through.

### B. Fix the Architecture
You cannot use the standard CellViT U-Net decoder with the current VirTues encoder output. You have two options:

**Option 1: The "Proper" Fix (Requires modifying VirTues code)**
*   Modify `FlexDualVirTuesEncoder.forward_list` to return intermediate states (e.g., output of every 3rd block).
*   Pass these as `z1`, `z2`, `z3` to the decoder.

**Option 2: The "Wrapper" Fix (Easier)**
*   **`z0` (Input):** Pass the DAPI channel (and maybe mean of proteins) directly to the decoder as `z0`. This restores high-res spatial information.
*   **`z1`-`z3`:** Since you don't have them, you might have to skip them or use simple CNN downsampling of `z0` to generate pseudo-skips.
*   **`z4` (Context):** Use the VirTues `ps` output here.

### C. Evaluation
*   **Metrics:** Continue with Dice.
*   **Visuals:** Always visualize the **DAPI channel** alongside your prediction. If you can't see DAPI in your visualization, your model can't see it either.

## 5. Summary of "What Went Wrong" in Phase 2
1.  **Blind Model:** You trained a segmentation model without the primary signal (DAPI).
2.  **Low-Res Vision:** You forced the model to segment 128x128 objects using only 16x16 feature maps, discarding all edge information.
3.  **Assumption:** You assumed `build_mm_datasets` would give you all necessary channels, but it was configured for a protein-only task.
