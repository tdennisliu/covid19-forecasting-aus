#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=10:00:00
#SBATCH --mem=30GB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dennis.liu01@adelaide.edu.au
#SBATCH --array=0-5

module load Python/3.6.1-foss-2016b
source $FASTDIR/virtualenvs/bin/activate

python model/contact_tracing/run_contact_tracing.py $SLURM_ARRAY_TASK_ID

deactivate
