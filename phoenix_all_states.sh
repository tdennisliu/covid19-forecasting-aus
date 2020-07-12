#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=1-12:00:00
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dennis.liu01@adelaide.edu.au
#SBATCH --array=0-7

module load Python/3.6.1-foss-2016b
source /fast/users/a1193089/virtualenvs/bin/activate

states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")

python run_state.py $1 $2 $3 ${states[$SLURM_ARRAY_TASK_ID]} $4


deactivate