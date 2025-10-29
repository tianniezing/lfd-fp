#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --array=0-127%15 # 128 jobs (8x2x2x2x2), max 15 run at once
#SBATCH --output=output2/slurm-%A_%a.out

module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0  # <-- ADD THIS LINE to load cuDNN
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/lfd3/bin/activate

learning_rates=(0.003 0.001)
optimizers=("Adam" "RMSprop")
batch_sizes=(16 32)
loss_functions=("binary_crossentropy")
dropouts=(0.3 0.5)
recurrent_dropouts=(0.1 0.3)
lstm_units=(128)
lstm_layers=(1 2)
bidirectional_flags=(True False)

num_lr=${#learning_rates[@]}
num_opt=${#optimizers[@]}
num_bs=${#batch_sizes[@]}
num_loss=${#loss_functions[@]}
num_do=${#dropouts[@]}
num_rdo=${#recurrent_dropouts[@]}
num_units=${#lstm_units[@]}
num_layers=${#lstm_layers[@]}
num_bidir=${#bidirectional_flags[@]}

total_jobs=$((num_lr * num_opt * num_bs * num_loss * num_do * num_rdo * num_units * num_layers * num_bidir))
echo "TOTAL COMBINATIONS TO RUN: $total_jobs"

task_id=$SLURM_ARRAY_TASK_ID

# Learning Rate
lr_idx=$((task_id % num_lr))
lr=${learning_rates[$lr_idx]}
task_id=$((task_id / num_lr))

# Optimizer
opt_idx=$((task_id % num_opt))
optimizer=${optimizers[$opt_idx]}
task_id=$((task_id / num_opt))

# Batch Size
bs_idx=$((task_id % num_bs))
batch_size=${batch_sizes[$bs_idx]}
task_id=$((task_id / num_bs))

loss_idx=$((task_id % num_loss))
loss=${loss_functions[$loss_idx]}
task_id=$((task_id / num_loss))

# Dropout
do_idx=$((task_id % num_do))
dropout=${dropouts[$do_idx]}
task_id=$((task_id / num_do))


rec_dropout=${recurrent_dropouts[$((task_id % num_rdo))]}
task_id=$((task_id / num_rdo))
units=${lstm_units[$((task_id % num_units))]}
task_id=$((task_id / num_units))
layers=${lstm_layers[$((task_id % num_layers))]}
task_id=$((task_id / num_layers))
bidir_setting=${bidirectional_flags[$((task_id % num_bidir))]}

if [ "$bidir_setting" = "True" ]; then
    bidir_arg="--bidirectional_layer"
else
    bidir_arg=""
fi

# --- Run the Experiment ---
echo "--- Starting Job $SLURM_ARRAY_TASK_ID/$total_jobs ---"
echo "Parameters: LR=$lr, OPT=$optimizer, BS=$batch_size, LOSS=$loss, DO=$dropout, RDO=$rec_dropout, UNITS=$units, LAYERS=$layers, BIDIR=$bidir_setting"

python ./lfd_fp_lstm.py \
    --embeddings /scratch/s4495845/LFD/FP/glove.twitter.27B.100d.txt \
    --learning_rate $lr \
    --optimizer $optimizer \
    --batch_size $batch_size \
    --loss_function $loss \
    --epochs 15 \
    --dropout $dropout \
    --recurrent_dropout $rec_dropout \
    --lstm_units $units \
    --lstm_layers $layers \
    $bidir_arg \
    --verbose 0

echo "--- Done with job $SLURM_ARRAY_TASK_ID ---"

deactivate