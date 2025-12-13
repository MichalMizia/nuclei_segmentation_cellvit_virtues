import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

# cellvit uses this, i think just to confuse us
norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


class CellViTWrapper:
    """Extract features from CellViT encoder at specific layers"""

    def __init__(self, cellvit_model, device="cuda"):
        """
        Args:
            cellvit_model: CellViT-256 model
            device: torch device string
        """
        self.model = cellvit_model
        self.encoder = cellvit_model.encoder
        self.device = device
        self.embeddings = {}
        self.embed_dim = self.encoder.embed_dim

        self.encoder.to(device)
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W) - should be normalized

        Returns:
            z1, z2, z3, z4: Intermediate features from encoder
                z1: Early features (B, embed_dim, H//16, W//16)
                z2: Mid features (B, embed_dim, H//16, W//16)
                z3: Late features (B, embed_dim, H//16, W//16)
                z4: Final PSS features (B, embed_dim, H//16, W//16)
        """
        patch_size = 16

        _, _, z = self.encoder(x)

        z0, z1, z2, z3, z4 = x, *z
        patch_dim = [int(d / patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        return z1, z2, z3, z4

    @torch.no_grad()
    def process_dataset(
        self,
        ds,
        return_intermediates: bool = True,
        max_batch_size: int = 32,
    ):
        """
        Process all tissues in dataset through CellViT encoder.

        Args:
            ds: MultimodalDataset with 'he' key
            return_intermediates: Whether to return intermediate layer features
            max_batch_size: Maximum number of patches to process at once

        Returns:
            Stores embeddings in self.embeddings dict with tissue_id as keys
        """
        patch_size = 64
        tissue_ids = ds.unimodal_datasets["he"].get_tissue_ids()
        tissue_ids = sorted(list(tissue_ids))

        tid = tissue_ids[0]
        C, H, W = ds.unimodal_datasets["he"].get_tissue(tid).shape

        print("=" * 60)
        print(f"CellViT Encoder - Processing {len(tissue_ids)} tissues".center(60))
        print(f"Image size: ({C}, {H}, {W})".center(60))
        print(f"Patch size: {patch_size}".center(60))
        print(f"Patches per image: {(H//patch_size) * (W//patch_size)}".center(60))
        print(f"Output features per patch: {(patch_size//16)**2}".center(60))
        print("=" * 60)

        for tid in tqdm(tissue_ids, desc="Processing tissues"):
            # Get normalized H&E image (3, H, W)
            he_image = ds.unimodal_datasets["he"]._get_tissue_all_channels(tid)
            he_image = torch.from_numpy(he_image).float() / 255.0  # (3, H, W)
            he_image = norm(he_image.to(self.device))

            he_patches = self._image_to_patches(he_image, patch_size)
            num_patches = len(he_patches)

            final_features = []
            intermediate_features = [[], [], []]  # z1, z2, z3

            for i in range(0, num_patches, max_batch_size):
                batch_patches = he_patches[i : min(i + max_batch_size, num_patches)]
                batch_tensor = torch.stack(batch_patches).cuda()  # (B, 3, 256, 256)

                z1, z2, z3, z4 = self.forward(batch_tensor)

                # z4 is PSS (final features): (B, embed_dim, 16, 16)
                # Flatten spatial dimensions: (B, embed_dim, 16, 16) -> (B*256, embed_dim)
                B, C, Hf, Wf = z4.shape
                final_features.append(
                    z4.flatten(2).transpose(1, 2).reshape(-1, C).cpu()
                )

                if return_intermediates:
                    # z1, z2, z3 are intermediate features: (B, embed_dim, 16, 16)
                    intermediate_features[0].append(
                        z1.flatten(2).transpose(1, 2).reshape(-1, C).cpu()
                    )
                    intermediate_features[1].append(
                        z2.flatten(2).transpose(1, 2).reshape(-1, C).cpu()
                    )
                    intermediate_features[2].append(
                        z3.flatten(2).transpose(1, 2).reshape(-1, C).cpu()
                    )

                del batch_tensor, z1, z2, z3, z4
                torch.cuda.empty_cache()

            # Concatenate all features
            final_output = torch.cat(final_features, dim=0)

            if return_intermediates:
                intermediate_outputs = [
                    torch.cat(feats, dim=0) for feats in intermediate_features
                ]
                self.embeddings[tid] = {
                    "pss": final_output,  # (N_total_features, embed_dim)
                    "intermediate_pss": intermediate_outputs,  # [(N, embed_dim), (N, embed_dim), (N, embed_dim)]
                }
            else:
                self.embeddings[tid] = {
                    "pss": final_output,
                }

    def save_embeddings(self, save_dir: str):
        """Save embeddings to disk"""
        os.makedirs(save_dir, exist_ok=True)
        for tid, data in self.embeddings.items():
            torch.save(data, os.path.join(save_dir, f"{tid}.pth"))
        print(f"Saved embeddings to {save_dir}")

    def load_embeddings(self, load_dir: str):
        """Load embeddings from disk"""
        self.embeddings = {}
        for fname in tqdm(sorted(os.listdir(load_dir)), desc="Loading embeddings"):
            if not fname.endswith(".pth"):
                continue
            tid = fname[:-4]
            self.embeddings[tid] = torch.load(
                os.path.join(load_dir, fname), map_location="cpu", weights_only=True
            )
        print(f"Loaded {len(self.embeddings)} tissue embeddings")

    def _image_to_patches(
        self, img: torch.Tensor, patch_size: int
    ) -> list[torch.Tensor]:
        """
        Split image into non-overlapping patches.

        Args:
            img: (C, H, W) tensor in range [0, 1] or [0, 255]
            patch_size: Size of square patches

        Returns:
            List of normalized (C, P, P) patches
        """
        C, H, W = img.shape
        P = patch_size

        # Ensure image is in [0, 1] range
        if img.max() > 1.0:
            img = img / 255.0

        if H % P != 0 or W % P != 0:
            new_H = H + (P - H % P) % P
            new_W = W + (P - W % P) % P
            img = T.Resize((new_H, new_W))(img.unsqueeze(0)).squeeze(0)

        # Use unfold to extract patches
        cols = F.unfold(img.unsqueeze(0), kernel_size=P, stride=P)  # (1, C*P*P, L)
        patches = cols.transpose(1, 2).reshape(-1, C, P, P)  # (L, C, P, P)

        return [p for p in patches]

    def _patches_to_image(
        self, patches: list[torch.Tensor], hw: tuple[int, int]
    ) -> torch.Tensor:
        """
        Reconstruct image from patches.

        Args:
            patches: List of (C, P, P) patches
            hw: Target (H, W) dimensions

        Returns:
            (C, H, W) reconstructed image
        """
        C, P, _ = patches[0].shape
        H, W = hw
        L = (H // P) * (W // P)

        if H % P != 0 or W % P != 0:
            raise ValueError(f"Target size ({H}, {W}) not divisible by patch_size={P}")

        assert len(patches) == L, f"Expected {L} patches, got {len(patches)}"

        stacked = torch.stack(patches)  # (L, C, P, P)
        cols = (
            stacked.reshape(L, C * P * P).transpose(0, 1).unsqueeze(0)
        )  # (1, C*P*P, L)
        img = F.fold(cols, output_size=(H, W), kernel_size=P, stride=P)  # (1, C, H, W)

        return img.squeeze(0)  # (C, H, W)
