#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --mem=2000
#SBATCH --array=0-9
#SBATCH --output=output2/slurm-%A_%a.out

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/lfd3_env/bin/activate

# Define parameter arrays (10 configurations)
learning_rates=(0.003 0.003 0.003 0.010 0.003 0.003 0.003 0.003 0.003 0.003)
optimizers=("Adam" "RMSprop" "RMSprop" "Adam" "Adam" "RMSprop" "Adam" "Adam" "Adam" "RMSprop")
batch_sizes=(64 32 16 64 16 32 16 64 64 16)
losses=("categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy" "categorical_crossentropy")
dropouts=(0.3 0.5 0.5 0.5 0.5 0.5 0.1 0.3 0.3 0.1)
recurrent_dropouts=(0.3 0.1 0.3 0.1 0.1 0.3 0.1 0.1 0.1 0.3)
lstm_units=(64 64 128 128 128 64 64 64 64 128)
lstm_layers=(2 3 2 2 2 1 3 3 2 2)
bidirectional=(True False True False False True False False False False)

# Select parameters for this job
lr=${learning_rates[$SLURM_ARRAY_TASK_ID]}
optimizer=${optimizers[$SLURM_ARRAY_TASK_ID]}
batch_size=${batch_sizes[$SLURM_ARRAY_TASK_ID]}
loss=${losses[$SLURM_ARRAY_TASK_ID]}
dropout=${dropouts[$SLURM_ARRAY_TASK_ID]}
rec_dropout=${recurrent_dropouts[$SLURM_ARRAY_TASK_ID]}
units=${lstm_units[$SLURM_ARRAY_TASK_ID]}
layers=${lstm_layers[$SLURM_ARRAY_TASK_ID]}
bidir=${bidirectional[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with:"
echo "lr=$lr, optimizer=$optimizer, batch_size=$batch_size, loss=$loss, dropout=$dropout, rec_dropout=$rec_dropout, units=$units, layers=$layers, bidirectional=$bidir"

$HOME/venvs/lfd3_env/bin/python ./lfd_assignment3_lstm.py \
    --learning_rate $lr \
    --loss_function $loss \
    --optimizer $optimizer \
    --batch_size $batch_size \
    --epochs 10 \
    --dropout $dropout \
    --recurrent_dropout $rec_dropout \
    --lstm_units $units \
    --lstm_layers $layers \
    --bidirectional_layer $bidir \
    --verbose 0

echo "Done with job $SLURM_ARRAY_TASK_ID"

deactivate
