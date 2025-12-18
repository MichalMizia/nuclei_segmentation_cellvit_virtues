#!/bin/bash
export RCP_CAAS_SCRATCH=course-cs-433-group01-scratch
export RCP_CAAS_GASPAR=your_gaspar
export RCP_CAAS_IMAGE=registry.rcp.epfl.ch/jon-docker/tissuevit:v1.1
export RCP_CAAS_PROJECT=course-cs-433-group01-your_gaspar
export RCP_CAAS_GPU=1 # num of GPUs

export RUNAI_UID=317575 # replace with your uid
export RUNAI_GID=86757

MSYS_NO_PATHCONV=1 runai submit \
    --image ${RCP_CAAS_IMAGE} \
    --name tissuevit \
    --gpu ${RCP_CAAS_GPU} \
    --node-pools default \
    --environment HOME="/home/${RCP_CAAS_GASPAR}" \
    --project ${RCP_CAAS_PROJECT} \
    --run-as-uid ${RUNAI_UID} \
    --run-as-gid ${RUNAI_GID} \
    --existing-pvc claimname=${RCP_CAAS_SCRATCH},path=/data \
    --existing-pvc claimname=home,path=/home/${RCP_CAAS_GASPAR} \
    --large-shm \
    --interactive \
    -- sleep infinity
