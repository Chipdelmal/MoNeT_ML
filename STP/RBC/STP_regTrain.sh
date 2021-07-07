#!/bin/bash
# chmod +x STP_clsTrain.sh

MTR=$1
PTH="./"
# Train models
python STP_regressTrain_br.py $MTR
python STP_regressTrain_rf.py $MTR
python STP_regressTrain_et.py $MTR