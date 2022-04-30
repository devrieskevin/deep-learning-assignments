#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

DATA_DIR="$HOME/assignment_1/code/cifar10/cifar-10-batches-py"
DNN="1000,1000"
MAX_STEP=5000
LR=0.0001
BS=500

# Run program
python ~/assignment_1/code/train_mlp_pytorch.py --data_dir $DATA_DIR --max_step $MAX_STEP --learning_rate $LR --batch_size $BS --dnn_hidden_units $DNN > "$TMPDIR"/mlp_pytorch_output.txt

#Copy output data from scratch to home
cp "$TMPDIR"/mlp_pytorch_output.txt ~/assignment_1/code/job_scripts/
cp "$TMPDIR"/mlp_pytorch.png ~/assignment_1/code/job_scripts/
