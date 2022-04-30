#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lnodes=1:ppn=1
#PBS -lmem=250GB

CODE_DIR="$HOME/assignment_2/part3"
DATA_DIR="$HOME/assignment_2/part3/assets"
DATA_FILE="ReZero_WN_TC_Arc4_ch1-30.txt"
MODEL_FILE="ReZero_model_state.pt"
OPTIM_FILE="ReZero_optimizer_state.pt"
SCHEDULER_FILE="ReZero_scheduler_state.pt"
LR=2e-3
EPOCHS=9
DROPOUT=0.9
LR_DECAY=0.96
LR_STEP=500
T=0.5

# Run program
python ~/assignment_2/part3/train.py --txt_file $DATA_DIR/$DATA_FILE --device cuda:0 --learning_rate $LR --epochs $EPOCHS --print_every 100 --sample_every 100 --example_len 100 --temperature $T --dropout_keep_prob $DROPOUT --learning_rate_decay $LR_DECAY --learning_rate_step $LR_STEP --model_state $CODE_DIR/$MODEL_FILE --optimizer_state $CODE_DIR/$OPTIM_FILE --scheduler_state $CODE_DIR/$SCHEDULER_FILE --load_state --save_state
