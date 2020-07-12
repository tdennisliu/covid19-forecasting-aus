states=("NSW" "VIC" "SA" "QLD" "TAS" "WA" "ACT" "NT")
for i in "${states[@]}"
do
    python run_state.py $1 $2 $3 $i ${4:-None}
done
python collate_states.py $1 $2 $3