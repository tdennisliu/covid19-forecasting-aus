dates=("1Apr" "8Apr" "15Apr" "22Apr" "29Apr" "6May" 
"13May" "20May" "27May" "03Jun" "10Jun" "17Jun" "24Jun" 
"01Jul" "08Jul" "15Jul" "22Jul" "29Jul" "05Aug" "12Aug" "19Aug" "26Aug" "02Sep" "16Sep" "23Sep" "30Sep" "07Oct")

for i in "${dates[@]}"
do
    python model/EpyReff/run_estimator.py $i
done
