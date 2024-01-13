#!/bin/bash
#
#SBATCH --job-name=SubTB
#SBATCH -p defq
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


cd ~/Projects/torchgfn_spinfoams/
source gfn_spinfoams/bin/activate
cd notebooks/
python Thanos_Generate_SubTB.py -p $SLURM_ARRAY_TASK_ID
