#!/bin/bash

DATE=$1
SHORTDATE=$2
NDAYS=$3

jid1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au phoenix_run_estimator.sh ${SHORTDATE})
echo $jid1

jid2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1 analysis/cprs/run_posteriors.sh ${DATE})
echo $jid2

jid3=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1,$jid2 phoenix_all_states.sh 200 ${NDAYS} ${DATE}) 
echo $jid3

jid4=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1,$jid2,$jid3 phoenix_collate_states.sh 200 ${NDAYS} ${DATE})

jid5=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1,$jid2,$jid3 phoenix_all_states.sh 5000 ${NDAYS} ${DATE})

jid6=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid5 phoenix_final_plots_csv.sh 5000 ${NDAYS} ${DATE})

