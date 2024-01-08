#!/bin/bash 

#SBATCH --mail-user=michael.koelle@ifi.lmu.de
#SBATCH --mail-type=FAIL
#SBATCH --partition=All
#SBATCH --export=NONE

# Use --partition=NvidiaAll if you need nodes with a gpu

# Set Environment Variables
export WANDB_MODE="disabled" # Use if you want to disable wandb
export WANDB_SILENT="true"

# virtualenv env
source env/bin/activate

# Runs the script
python src/main.py $@
