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
#SBATCH --array=1-100


paramFile="./model/contact_tracing/inputs.csv"

module load arch/haswell
module load Python/3.6.1-foss-2016b
source ../virtualenvs/bin/activate

echo "array_job_index: $SLURM_ARRAY_TASK_ID"

i=1
found=0 

while IFS=, read a b c d
do 
    if [ $i = $SLURM_ARRAY_TASK_ID ]; then
        echo "Running $simName with [$a, $b, $c, $d]"
        found=1 

        break 
    fi 
    i=$((i + 1)) 
done < $paramFile

if [ $found = 1 ]; then
    python model/contact_tracing/run_contact_tracing.py $a $b $c $d $1
else 
  echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is outside range of input file $paramFile" 
fi


deactivate
