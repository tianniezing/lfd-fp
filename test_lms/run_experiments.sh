#!/bin/bash
#SBATCH --job-name=offensive_transformer_search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/LFD/FP/lms/slurm_logs/slurm-%A_%a.out
#SBATCH --array=0-161

set -e 
set -x
mkdir -p /scratch/s4495845/LFD/FP/lms/slurm_logs

module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/venvs/lfd3/bin/activate

MODELS=("microsoft/deberta-base-mnli" "bert-base-uncased" "roberta-base")
LEARNING_RATES=(1e-5 2e-5 3e-5)
BATCH_SIZES=(8 16 32)
MAX_LENGTHS=(64 128 256)
PATIENCE_VALUES=(2 3)

NUM_MODELS=${#MODELS[@]}
NUM_LRS=${#LEARNING_RATES[@]}
NUM_BATCH_SIZES=${#BATCH_SIZES[@]}
NUM_MAX_LENGTHS=${#MAX_LENGTHS[@]}
NUM_PATIENCE=${#PATIENCE_VALUES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID
PAT_IDX=$((TASK_ID % NUM_PATIENCE))
LEN_IDX=$(((TASK_ID / NUM_PATIENCE) % NUM_MAX_LENGTHS))
BS_IDX=$(((TASK_ID / (NUM_PATIENCE * NUM_MAX_LENGTHS)) % NUM_BATCH_SIZES))
LR_IDX=$(((TASK_ID / (NUM_PATIENCE * NUM_MAX_LENGTHS * NUM_BATCH_SIZES)) % NUM_LRS))
MODEL_IDX=$(((TASK_ID / (NUM_PATIENCE * NUM_MAX_LENGTHS * NUM_BATCH_SIZES * NUM_LRS)) % NUM_MODELS))

MODEL=${MODELS[$MODEL_IDX]}
LR=${LEARNING_RATES[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BS_IDX]}
MAX_LEN=${MAX_LENGTHS[$LEN_IDX]}
PATIENCE=${PATIENCE_VALUES[$PAT_IDX]}

MODEL_NAME_SAFE=$(echo "$MODEL" | tr '/' '_')
OUTPUT_DIR="experiments/run_${TASK_ID}__${MODEL_NAME_SAFE}"
mkdir -p $OUTPUT_DIR
LOG_FILE="${OUTPUT_DIR}/summary.log"

echo "Experiment Summary" > $LOG_FILE
echo "RUN_ID: ${TASK_ID}" | tee -a $LOG_FILE
echo "MODEL: ${MODEL}" | tee -a $LOG_FILE
echo "LEARNING_RATE: ${LR}" | tee -a $LOG_FILE
echo "BATCH_SIZE: ${BATCH_SIZE}" | tee -a $LOG_FILE
echo "MAX_LENGTH: ${MAX_LEN}" | tee -a $LOG_FILE
echo "PATIENCE: ${PATIENCE}" | tee -a $LOG_FILE
echo "-------------------------" | tee -a $LOG_FILE

python transformer_trainer.py \
    --model_name "$MODEL" \
    --train_file "train.tsv" \
    --dev_file "dev.tsv" \
    --test_file "test.tsv" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 15 \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_length $MAX_LEN \
    --early_stopping_patience $PATIENCE \
    | tee -a $LOG_FILE

echo "Job ${TASK_ID} finished." | tee -a $LOG_FILE