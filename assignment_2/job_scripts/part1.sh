#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lnodes=1:ppn=1
#PBS -lmem=250GB

CODE_DIR="$HOME/deep_learning/assignments/assignment_2/part1"
DEVICE="cpu"

mkdir $CODE_DIR/output

# Run program
for model in "RNN" "LSTM"; do
    for t in $(seq 5 20); do
        NAME="${model}_${t}"
        python $CODE_DIR/train.py --device $DEVICE --model_type $model --input_length $t \
        --summary_path $CODE_DIR/summaries/ --summary_name $NAME > $CODE_DIR/output/$NAME

    done
done

