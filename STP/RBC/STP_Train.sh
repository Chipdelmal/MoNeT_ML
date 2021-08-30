#!/bin/bash
# chmod +x STP_clsTrain.sh

SET=$1
MTR=$2
PTH=$3
# Train models
python STP_Train_b.py $SET $MTR $PTH
python STP_Train_rf.py $SET $MTR $PTH
python STP_Train_et.py $SET $MTR $PTH