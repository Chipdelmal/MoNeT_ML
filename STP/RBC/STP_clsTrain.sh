#!/bin/bash
# chmod +x STP_clsTrain.sh

MTR=$1
PTH="./"
# Train models
python STP_clsTrain_bc.py $MTR
python STP_clsTrain_rf.py $MTR
python STP_clsTrain_et.py $MTR