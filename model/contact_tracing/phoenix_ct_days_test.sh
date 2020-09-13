#!/bin/bash
#SBATCH -p batch
#SBATCH --qos=express
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=1-00:00:00
#SBATCH --mem=70GB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laura.boyle@adelaide.edu.au
#SBATCH --array=0-5

module load Python/3.6.1-foss-2016b
source fast/users/a1226521/testvirtualenv/bin/activate

