FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Basic System Dependencies
RUN apt-get update \
    && apt-get install -y sudo wget rsync rclone git tmux build-essential cmake dssp zip nano unzip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda
USER root
RUN apt update \
    && apt install wget git -y \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists
ENV PATH=/opt/conda/bin:$PATH
RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh -q && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# 3. Create Conda Env (Standard Path)
# We copy the files to a temporary location first just to build the env
ARG ENV_NAME="tissuevit"
COPY environment.yaml /tmp/environment.yaml
COPY requirements_dinov2.txt /tmp/requirements_dinov2.txt

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
RUN conda env create --yes -f /tmp/environment.yaml -n ${ENV_NAME}

# 4. Setup Bash Activation
RUN echo -n ". /opt/conda/etc/profile.d/conda.sh\nconda activate ${ENV_NAME}\n" >> /root/.bashrc && \
    echo -n ". /opt/conda/etc/profile.d/conda.sh\nconda activate ${ENV_NAME}\n" >> /etc/skel/.bashrc
SHELL ["/bin/bash", "-ic"]

# --- CRITICAL FIX: SETUP WORKSPACE FIRST ---
# In your original file, you set WORKDIR before copying the requirements for the pip install step.
# We do the same here so 'pip install -r' finds the file.
WORKDIR /workspace
ENV PATH="/workspace/bin:$PATH"

# Copy the requirements file INTO the workspace
COPY requirements_dinov2.txt .

# 5. Install Pip Packages
# (Exact list from your original file)
RUN conda run -n tissuevit pip install numpy
RUN conda run -n tissuevit pip install pandas
RUN conda run -n tissuevit pip install einops
RUN conda run -n tissuevit pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN conda run -n tissuevit pip install biopython
RUN conda run -n tissuevit pip install h5py
RUN conda run -n tissuevit pip install fair-esm
RUN conda run -n tissuevit pip install readimc
RUN conda run -n tissuevit pip3 install -U scikit-learn
RUN conda run -n tissuevit pip install -U matplotlib
RUN conda run -n tissuevit pip install seaborn
RUN conda run -n tissuevit pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
RUN conda run -n tissuevit pip install wandb
RUN conda run -n tissuevit pip install tifffile
RUN conda run -n tissuevit pip install pillow
RUN conda run -n tissuevit pip install umap-learn
RUN conda run -n tissuevit pip install ipython
RUN conda run -n tissuevit pip install notebook
RUN conda run -n tissuevit pip install ipywidgets
RUN conda run -n tissuevit pip install POT
RUN conda run -n tissuevit pip install loguru
RUN conda run -n tissuevit pip install transformers
RUN conda run -n tissuevit pip install multimil
RUN conda run -n tissuevit pip install imbalanced-learn
RUN conda run -n tissuevit pip install timm
RUN conda run -n tissuevit pip install omegaconf
# This will now work because requirements_dinov2.txt is in /workspace
RUN conda run -n tissuevit pip install -r requirements_dinov2.txt
RUN conda run -n tissuevit pip install flash-attn --no-build-isolation
RUN conda run -n tissuevit pip install scikit-image
# RUN conda run -n tissuevit pip install torch-scatter

# --- CRITICAL FOR TEAM: OPEN PERMISSIONS ---
USER root
RUN chmod 777 -R /opt/conda

CMD ["/bin/bash"]