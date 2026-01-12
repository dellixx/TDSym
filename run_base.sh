#!/bin/bash


GPU_ID=0
LOG_DIR="logs_base"

# Experiment Scope
DATASETS=("ICEWS14")
MODELS=("TeLM" "TeAST" "TNTComplEx" "TComplEx" "TCompoundE" "TeRDy")
RANKS=(32 64)  

# Setup Environment
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "============================================"
echo "🚀 Starting Experiments on GPU: $GPU_ID"
echo "============================================"

# 2. Training Loop
# -------------------------
for dataset in "${DATASETS[@]}"; do
    
    # Check if dataset exists to avoid errors
    if [ ! -d "data/$dataset" ]; then
        echo "⚠️  [Skip] Data folder not found: data/$dataset"
        continue
    fi

    for model in "${MODELS[@]}"; do
        for rank in "${RANKS[@]}"; do
            
            # Define specific log file for this run
            LOG_FILE="${LOG_DIR}/${dataset}_${model}_rank${rank}.log"
            
            echo "----------------------------------------------------"
            echo "▶️  Running: Dataset=[$dataset] | Model=[$model] | Rank=[$rank]"
            echo "   📄 Log saved to: $LOG_FILE"
            
            # Write start time to log
            echo "Experiment Start: $(date)" > "$LOG_FILE"
            
            # Execute Training
            python learner.py \
                --dataset "$dataset" \
                --model "$model" \
                --rank "$rank" \
                >> "$LOG_FILE" 2>&1
                
        done
    done
done

echo "============================================"
echo "🎉 All Experiments Completed!"
echo "============================================"