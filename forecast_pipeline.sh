#!/bin/bash

DATE=$1
NDAYS=$2

jid1=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au phoenix_run_estimator.sh ${DATE})

jid2=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1 analysis/cprs/run_posteriors.sh ${DATE})


jid4=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid1,$jid2 phoenix_all_states.sh 20000 ${NDAYS} ${DATE})

jid5=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid4 phoenix_final_plots_csv.sh 20000 ${NDAYS} ${DATE})
echo "Normal Run:", $jid4, $jid5


jid6=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid5 phoenix_all_states.sh 20000 ${NDAYS} ${DATE} None UK)

jid7=$(sbatch --parsable --mail-user=$USER@adelaide.edu.au --dependency=afterok:$jid6 phoenix_final_plots_csv.sh 20000 ${NDAYS} ${DATE} UK)
echo "VoC Run:", $jid6, $jid7
