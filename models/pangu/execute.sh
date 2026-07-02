#!/bin/bash

# Required — set these to your local paths
MODEL_PATH=${MODEL_PATH:-"pangu_weather_24.onnx"}
DATA_PATH=${DATA_PATH:-"path/to/era5_data.nc"}

# Optional parameters
PRIOR_DIM=${PRIOR_DIM:-"[1,2]"}
REG_PARAM=${REG_PARAM:-1e8}
JOB_NAME=${JOB_NAME:-"PanguDA_default"}
KNOWN_ATM=${KNOWN_ATM:-"None"}
LBFGSSTEP=${LBFGSSTEP:-15}

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

uv run python3 prio_knowledge_training_single_gpu.py \
    --model_path="${MODEL_PATH}" \
    --data_path="${DATA_PATH}" \
    --prior_known_dim="${PRIOR_DIM}" \
    --known_atmosphere="${KNOWN_ATM}" \
    --reg_param ${REG_PARAM} \
    --name="${JOB_NAME}" \
    --LBFGSsteps=${LBFGSSTEP} \
    --git_commit "${GIT_COMMIT}"
