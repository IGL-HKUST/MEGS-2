#!/bin/bash

#  |   Scenes  | sharpness_ratio | Pruning_ratio1 | Pruning_ratio2 |
#  |:---------:|:---------------:|:--------------:|:--------------:|
#  |  bicycle  |        65       |       75       |       65       |
#  |   stump   |        68       |       55       |       68       |
#  |   garden  |        55       |       60       |       55       |
#  |  flowers  |        68       |       50       |       68       |
#  |  treehill |        68       |       55       |       68       |
#  |   bonsai  |        68       |       55       |       68       |
#  |  counter  |        68       |       55       |       68       |
#  |  kitchen  |        68       |       50       |       68       |
#  |    room   |        70       |       60       |       70       |
#  |  playroom |        65       |       50       |       65       |
#  | drjohnson |        65       |       55       |       65       |
#  |   train   |        68       |       50       |       68       |
#  |   trunk   |        68       |       50       |       68       |

# Path to your Python script
PYTHON_SCRIPT="./train.py"  
PYTHON_SCRIPT_RENDER="./render.py"  
PYTHON_SCRIPT_METRICS="./metrics.py"
PYTHON_SCRIPT_MEM="./render_mem_test.py"
BASE_DATASET_DIR="../dataset/mip360"
# BASE_DATASET_DIR="../dataset/tandt"
# BASE_DATASET_DIR="../dataset/db"
chkpnt_iter=14999
declare -a run_scenes=(
  "bicycle"
  # "stump"
  # "garden"
  # "treehill"
  # "flowers"
  # "bonsai"
  # "counter"
  # "kitchen"
  # "room"
  # "train"
  # "truck"
  # "playroom"
  # "drjohnson"
)


PORT="1242"
SPA_INTERVAL="50"

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=2000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -v threshold="$mem_threshold" -F', ' '
      $2 < threshold { print $1; exit }
    '
}


run_script(){
  while ss -tuln | grep -q ":$PORT";do
    echo "Port $PORT is in use."
    PORT=$((PORT + 1))
    echo "New port number is $PORT"
  done
danshi
  local DATASET_DIR=$1
  local DATASET_NAME=$(basename "$DATASET_DIR")
  local SPA_RATIO1=0.55
  local SPA_RATIO2=0.65
  local sharpness_ratio=0.65
  local sharpness_threshold=1
  OUTPUT_DIR="./output/"$DATASET_NAME"/"$SPA_RATIO1"_"$SPA_RATIO2"_"$sharpness_threshold"_"$sharpness_ratio""
  echo "Output script for $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  ckpt="$OUTPUT_DIR"/chkpnt"$chkpnt_iter".pth
  if [ -f "$OUTPUT_DIR/$OUTPUT_FILE" ]; then
      echo "Output file $OUTPUT_FILE already exists. Skipping this iteration."
      continue
  fi

  gpu_id=$(get_available_gpu)
  if [[ -n $gpu_id ]]; then
    echo "GPU $gpu_id is available."
    CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT" \
    --port "$PORT" \
    -s="$DATASET_DIR" \
    -m="$OUTPUT_DIR" \
    --eval \
    --prune_ratio1 "$SPA_RATIO1"\
    --prune_ratio2 "$SPA_RATIO2"\
    --sharpness_threshold "$sharpness_threshold"\
    --sharpness_ratio "$sharpness_ratio"\
    --imp_metric "outdoor" \
    -i images_4
    #--start_checkpoint "$ckpt"

    CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT_RENDER" \
    -m="$OUTPUT_DIR" \
    --skip_train 
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT_METRICS" \
    -m="$OUTPUT_DIR" 
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT_MEM" \
    -m="$OUTPUT_DIR"
    else
      echo "No GPU available at the moment. Retrying in 1 minute."
      sleep 60
  fi
}

for view in "${run_scenes[@]}"; do
    echo "Running script for $view"
    run_script "$BASE_DATASET_DIR/$view"
done