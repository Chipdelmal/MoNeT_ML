#!/bin/bash
# chmod +x STP_clsTrain.sh

SET=$1
MTR=$2
PTH="./"
# Train models
python STP_Train_b.py $SET $MTR
python STP_Train_rf.py $SET $MTR
python STP_Train_et.py $SET $MTR