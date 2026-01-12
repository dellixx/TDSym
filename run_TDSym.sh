#!/bin/bash

GPU_ID=0
LOG_DIR="logs_TDSym"

# Target Model
MODEL_NAME="TDSym"

# Datasets to run
DATASETS=("ICEWS14")

# Rank configurations
RANKS=(32 64)

# 2. Environment Setup
# -------------------------
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================"
echo "🚀 Starting Experiments for model: $MODEL_NAME"
echo "============================================"

# 3. Training Loop
# -------------------------
for dataset in "${DATASETS[@]}"; do
    
    # Check if dataset directory exists
    if [ ! -d "data/$dataset" ]; then
        echo "⚠️  [Skip] Data folder not found: data/$dataset"
        continue
    fi

    for rank in "${RANKS[@]}"; do
        
        # Define log file path
        LOG_FILE="${LOG_DIR}/${dataset}_${MODEL_NAME}_rank${rank}.log"
        
        echo "----------------------------------------------------"
        echo "▶️  Running: Dataset=[$dataset] | Rank=[$rank]"
        echo "   📄 Log saved to: $LOG_FILE"
        
        # Initialize log with timestamp
        echo "Experiment Start: $(date)" > "$LOG_FILE"

        # Execute training command
        python learner.py \
            --dataset "$dataset" \
            --model "$MODEL_NAME" \
            --rank "$rank" \
            >> "$LOG_FILE" 2>&1
            
    done
done

echo "============================================"
echo "🎉 All Experiments Completed!"
echo "============================================"