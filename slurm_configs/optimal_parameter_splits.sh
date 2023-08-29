# for one_hot, esm, transformer representation

OPTIMAL_RF_PARAM_FILENAME_REFERENCE="RF_optimal_estimators_1FQG_transformer_RandomSplitter_None_s"
OPTIMAL_RF_PARAM_FILENAME="RF_optimal_estimators_1FQG_transformer_FractionalRandomSplitter"
FRACTIONS=("0.45" "0.51" "0.54" "0.57" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.85" "0.9" "0.95" "1.0")

for s in ${FRACTIONS[@]}; do
    for split in $(seq 1 4); do
        cp $OPTIMAL_RF_PARAM_FILENAME_REFERENCE${split}".pkl" $OPTIMAL_RF_PARAM_FILENAME"_"${s}"_None_s"${split}".pkl"
        # echo $OPTIMAL_RF_PARAM_FILENAME_REFERENCE${split}".pkl"
        # echo $OPTIMAL_RF_PARAM_FILENAME"_"${s}"_None_s"${split}".pkl"
    done
done