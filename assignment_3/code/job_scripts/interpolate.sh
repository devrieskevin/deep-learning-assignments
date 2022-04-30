#!/bin/sh
#PBS -lwalltime=00:05:00
#PBS -lnodes=1:ppn=1
#PBS -lmem=250GB

CODE_DIR="$HOME"
DEVICE="cuda:0"

# Run program
python $CODE_DIR/a3_gan_interpolation.py --device $DEVICE
