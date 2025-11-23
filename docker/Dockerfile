FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
# Basic stuff
RUN apt-get update \
    && apt-get install -y sudo wget rsync rclone git tmux build-essential cmake dssp zip nano unzip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
## Install miniconda ##
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
## Create conda env ##
# env name can be customized on build -> --build-arg ENV_NAME=my-custom-env-name
ARG ENV_NAME="tissuevit"
COPY environment.yaml .
COPY requirements_dinov2.txt .
# Create conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create --yes -f environment.yaml -n ${ENV_NAME}
# Add conda env activation in root home and home skel (https://www.linux.com/training-tutorials/users-groups-and-other-linux-beasts)
RUN echo -n ". /opt/conda/etc/profile.d/conda.sh\nconda activate ${ENV_NAME}\n" >> /root/.bashrc && \
    echo -n ". /opt/conda/etc/profile.d/conda.sh\nconda activate ${ENV_NAME}\n" >> /etc/skel/.bashrc
# Make RUN commands to read bashrc to activate conda
SHELL ["/bin/bash", "-ic"]
#RUN ln -s /usr/bin/mkdssp /usr/bin/dssp

# Build arguments. Adjust these to match your LDAP settings. You should be able to see them when typing "id" on the jumphost.rcp.epfl.ch
ARG LDAP_USERNAME="kuci"
ARG LDAP_UID=308525
ARG LDAP_GROUPNAME="kuci"
ARG LDAP_GID=30132
ARG LDAP_SUPPGROUPNAME="rcp-runai-course-cs-433-group01_AppGrpU"
ARG LDAP_SUPPGROUPGID=86757

# Create your user in the container
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN groupadd ${LDAP_SUPPGROUPNAME} --gid ${LDAP_SUPPGROUPGID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} -G ${LDAP_SUPPGROUPNAME} ${LDAP_USERNAME}
# Switch to the local user
USER ${LDAP_USERNAME}
# Specify paths
ARG WORKDIR="/home/${LDAP_USERNAME}"
WORKDIR $WORKDIR
RUN mkdir -p $WORKDIR
RUN mkdir -p $WORKDIR/bin
ENV PATH="$WORKDIR/bin:$PATH"
COPY requirements_dinov2.txt .
#ENV PATH="$WORKDIR/bin/miniconda3/bin:$PATH"
# Perform uninstallations and installations within the tissuevit environment
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
RUN conda run -n tissuevit pip install -r requirements_dinov2.txt
RUN conda run -n tissuevit pip install flash-attn --no-build-isolation
RUN conda run -n tissuevit pip install scikit-image
RUN conda run -n tissuevit pip install torch-scatter
USER root 
RUN chmod 777 -R /opt/conda/envs/tissuevit
USER ${LDAP_USERNAME}
# Ensure tissuevit is activated for any further commands
#RUN echo "conda activate tissuevit" >> ~/.bashrc
# Reset SHELL to default if necessary for further commands
#SHELL ["/bin/bash", "-c"]
