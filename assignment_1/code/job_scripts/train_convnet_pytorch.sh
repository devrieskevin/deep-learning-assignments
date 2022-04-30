#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

DATA_DIR="$HOME/assignment_1/code/cifar10/cifar-10-batches-py"
LR=0.001

# Run program
python ~/assignment_1/code/train_convnet_pytorch.py --data_dir $DATA_DIR > "$TMPDIR"/convnet_pytorch_output.txt

#Copy output data from scratch to home
cp "$TMPDIR"/convnet_pytorch_output.txt ~/assignment_1/code/job_scripts/
cp "$TMPDIR"/convnet_pytorch.png ~/assignment_1/code/job_scripts/
