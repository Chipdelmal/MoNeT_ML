# Dev Notes

## To think about and implement

* Use the CLS dataset for classification and the REG for regression (do not concatenate)
* REG will probably need "one-hot" encoding for the SEX feature
* Check that the methods used for the ensembles should be subject to feature scaling and/or normalization
* Use `sys.argv` for automation
* Think about creating a wrapper script for classification and another one for regression (most of the code is the same across methods, the only changes are around the model used)


## Changes

* `STP_regressTrain_br.py` contains the changes to use for the scripts
  * Launch with: `python STP_regressTrain_br.py 'CPT'`
* Created an `STP_constants` file to store all the constants that are common to the algorithms
* Created a template for the bash script

## Future

* Use some paths to read data and store results
