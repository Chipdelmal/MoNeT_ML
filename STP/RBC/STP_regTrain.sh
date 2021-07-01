#!/bin/bash
# chmod +x STP_clsTrain.sh

MTR=$1
PTH="./"
# Train models
python STP_regressTrain_br.py $MTR