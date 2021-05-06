#!/bin/bash

# argv1: USR
# argv2: LND

MTR="WOP"
VT_SPLIT="0.25"
KFOLD="10"

declare -a quantiles=("50" "75" "90")
declare -a metrics=("POE" "WOP")

for MTR in ${metrics[@]}; do
   for QNT in ${quantiles[@]}; do
      python PYF_Preprocess.py $1 $2 $MTR $QNT
   done
done


python PYF_Train.py $1 $2 $MTR $QNT $VT_SPLIT $KFOLD