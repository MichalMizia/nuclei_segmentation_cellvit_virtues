# 1. Base Image
FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
ARG TARGETPLATFORM=linux/amd64
ENV DEBIAN_FRONTEND=noninteractive

# 2. System Dependencies
RUN apt-get update && apt-get install -y \
    sudo wget rsync rclone git tmux build-essential cmake dssp zip nano unzip ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Miniconda
USER root
ENV PATH=/opt/conda/bin:$PATH
RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh -q && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# 4. Create Conda Environment
ARG ENV_NAME="tissuevit"
COPY environment.yaml /tmp/environment.yaml
COPY requirements_dinov2.txt /tmp/requirements_dinov2.txt

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
RUN conda env create --yes -f /tmp/environment.yaml -n ${ENV_NAME}

# Setup Shell to use Conda automatically
RUN echo "source /opt/conda/bin/activate ${ENV_NAME}" >> /root/.bashrc
SHELL ["/bin/bash", "-c"]

# 5. Workspace Setup
WORKDIR /workspace
ENV PATH="/workspace/bin:$PATH"
COPY requirements_dinov2.txt .

# --- SAFE INSTALLATION BLOCKS ---

# BLOCK A: Install PyTorch & Xformers (The Heavy Downloads)
# We do this first because it takes the longest and changes the least.
RUN source /opt/conda/bin/activate tissuevit && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu124

# BLOCK B: Install Standard Libraries (Fast)
# These are standard packages that rarely fail.
RUN source /opt/conda/bin/activate tissuevit && \
    pip install --no-cache-dir \
    numpy pandas einops biopython h5py fair-esm readimc \
    scikit-learn matplotlib seaborn wandb tifffile pillow \
    umap-learn ipython notebook ipywidgets POT loguru transformers \
    multimil imbalanced-learn timm omegaconf scikit-image

# BLOCK C: Install The "Problem Children" (Slow/Complex)
# flash-attn compiles from source and often breaks memory limits. We isolate it.
# We also run the requirements file here.
RUN source /opt/conda/bin/activate tissuevit && \
    pip install --no-cache-dir flash-attn --no-build-isolation && \
    pip install --no-cache-dir -r requirements_dinov2.txt

# 6. Final Permissions
USER root
RUN chmod 777 -R /opt/conda

CMD ["/bin/bash"]