                 ┌────────────────────┐
                 │  DATA SOURCES      │
                 ├────────────────────┤
                 │ H&E slides         │  (Phase 1)
                 │ SP CycIF channels  │  (Phase 2)
                 │ Segmentation masks │  (Phase 2)
                 └─────────┬──────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   DATA LOADER        │
                │ (already provided)   │
                ├──────────────────────┤
                │ he_base.py           │
                │ mm_base.py           │
                │ multiplex_base.py    │
                └──────────┬───────────┘
                           │
                           ▼
         ┌──────────────────────────────────────────┐
         │             ENCODER STAGE                │
         ├──────────────────────────────────────────┤
         │ 1. CellViT default encoder (ViT)         │
         │        (Phase 1 – already implemented)   │
         │                                           │
         │ 2. VirTues encoder                       │
         │    (Phase 2 – already provided to you)   │
         │    flex_dual_virtues.py                  │
         └───────────┬──────────────┬──────────────┘
                     │              │
                     │              │ Channel Tokens
                     │              └───────────────▶ [C tokens]
                     │
                     │ PSS Tokens
                     └─────────────────────────────▶ [Global token]
                     
                           ▼
           ┌─────────────────────────────────┐
           │     CELLViT DECODER HEAD        │
           ├─────────────────────────────────┤
           │  This is the segmentation head  │
           │  (Already implemented in CellViT)│
           │                                   │
           │  You DO NOT write this yourself   │
           └─────────────────┬─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  MASK OUTPUT    │
                   ├─────────────────┤
                   │ Semantic mask   │
                   │ (cell type per pixel) │
                   └─────────────────┘

