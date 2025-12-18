#!/bin/bash

export RCP_CAAS_SCRATCH=course-cs-433-group01-scratch
export RCP_CAAS_GASPAR=your_gaspar
export RCP_CAAS_IMAGE=registry.rcp.epfl.ch/jon-docker/tissuevit:v1.1
export RCP_CAAS_PROJECT=course-cs-433-group01-your_gaspar
export RCP_CAAS_GPU=1

export RUNAI_UID=317575 # replace with your UID
export RUNAI_GID=86757

TARGET_DIR="/home/mizia/project-2-gradient_tri_scent/notebooks" # the directory containing the script to run
SCRIPT_NAME="criterion_experiment.py"
TARGET_SCRIPT="${TARGET_DIR}/${SCRIPT_NAME}"
JOB_NAME="criterion-experiment"

echo "================================================"
echo "Submitting experiment job: ${JOB_NAME}"
echo "Target location: ${TARGET_SCRIPT}"
echo "================================================"

# Delete old job if exists
runai delete job ${JOB_NAME} -p ${RCP_CAAS_PROJECT} 2>/dev/null || true
sleep 2

# Submit the job
MSYS_NO_PATHCONV=1 runai submit \
    --image ${RCP_CAAS_IMAGE} \
    --name ${JOB_NAME} \
    --gpu ${RCP_CAAS_GPU} \
    --node-pools h100 \
    --environment HOME="/home/${RCP_CAAS_GASPAR}" \
    --project ${RCP_CAAS_PROJECT} \
    --run-as-uid ${RUNAI_UID} \
    --run-as-gid ${RUNAI_GID} \
    --existing-pvc claimname=${RCP_CAAS_SCRATCH},path=/data \
    --existing-pvc claimname=home,path=/home/${RCP_CAAS_GASPAR} \
    --large-shm \
    --command -- /bin/bash -c '
        set -ex
        
        echo "========================================="
        echo "Job started at: $(date)"
        echo "========================================="
        
        # Activate conda environment
        echo "=== Activating conda environment ==="
        source /opt/conda/bin/activate tissuevit
        echo "Active environment: $CONDA_DEFAULT_ENV"
        
        # Show Python info
        which python
        python --version
        
        # Reinstall PyTorch and dependencies
        echo "=== Reinstalling PyTorch 2.5.1 and dependencies ==="
        pip uninstall -y flash-attn torch torchvision torchaudio xformers || true
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --extra-index-url https://download.pytorch.org/whl/cu124
        pip install flash-attn --no-build-isolation --no-cache-dir
        pip install kornia cellvit
        pip install --upgrade setuptools
        
        # Navigate to experiment directory
        echo "=== Navigating to experiment directory ==="
        cd '"${TARGET_DIR}"'
        pwd
        
        # Run experiment
        echo "========================================="
        echo "Starting experiment at: $(date)"
        echo "========================================="
        
        python "${SCRIPT_NAME}" 2>&1
        RESULT=$?
        
        echo "========================================="
        echo "Experiment completed at: $(date)"
        echo "Exit code: ${RESULT}"
        echo "========================================="
        
        if [ ${RESULT} -eq 0 ]; then
            echo "=== Experiment finished successfully ==="
        else
            echo "=== Experiment failed with errors ==="
            exit ${RESULT}
        fi
    '

if [ $? -eq 0 ]; then
    echo ""
    echo "Job submitted successfully!"
    echo ""
    echo "Monitor with:"
    echo "  runai describe job ${JOB_NAME} -p ${RCP_CAAS_PROJECT}"
    echo "  runai logs ${JOB_NAME} -p ${RCP_CAAS_PROJECT} --follow"
else
    echo ""
    echo "ERROR: Failed to submit job"
fi
