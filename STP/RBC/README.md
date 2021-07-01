# São Tomé and Príncipe

## Datasets

* `CLS_HLT_50Q_10T.csv`
* `REG_HLT_50Q_10T.csv`
* `SCA_HLT_50Q_10T.csv`

These datasets are then concatentated into one Pandas Dataframe for the classifiers/regressors to train and test on.

## Python Scripts Nomenclature

`task` represents the type of machine learning task used on the datasets (either Classifier or Regressor).

`model` represents which model used to execute the task on the datasets (e.g. using an Extra Trees model)

* `STP_taskTrain_model.py`
    * Replace `task` with: 
        * `cls`: Classifier
        * `regress`: Regressor
    * Replace `model` with: 
        * `bc`: Bagging Classifier
        * `br`: Bagging Regressor
        * `et`: Extra Trees
        * `gbt`: Gradient Boosted Trees
        * `rf`: Random Forest
        * `sc`: Stacking Classifier
        * `sr`: Stacking Regressor
        * `vc`: Voting Classifier
        * `vr`: Voting Regressor

## Running Classifiers/Regressors

To run a classifier or regressor on the datasets, simply use the `py` or `python3` command on the script of your choice

e.g. to run a Random Forest Regressor, run:

```bash
py STP_regressTrain_rf.py
```

## Result .txt Files

The above python scripts generate a report `.txt` file that shows meta-information such as the features/input variables and summary statistics of the model used such as a R2 score for the regressors.

## .txt Files Nomenclature

`task` represents the type of machine learning task used on the datasets (either Classifier or Regressor).

`model` represents which model used to execute the task on the datasets (e.g. using an Extra Trees model)

`metric` represents the outcome variable the model was tasked to fit on.

* `task_model_metric.py`
    * Replace `task` with: 
        * `CLS`: Classifier
        * `REG`: Regressor
    * Replace `model` with: 
        * `BC`: Bagging Classifier
        * `BR`: Bagging Regressor
        * `ET`: Extra Trees
        * `GBT`: Gradient Boosted Trees
        * `RF`: Random Forest
        * `SC`: Stacking Classifier
        * `SR`: Stacking Regressor
        * `VC`: Voting Classifier
        * `VR`: Voting Regressor
    * Replace `metric` with:
        * `CPT`: Cumulative fraction of mosquitoes divided by time
        * `TTI`: Time to introgression
        * `TTO`: Time to outrogression
        * `WOP`: Window of protection
        * `POE`: Probability of elimination/fixation
        * `MIN`: Minimum of mosquitoes
        * `RAP`: Fraction of mosquites at timepoint

## Example Code

Much of the code for the python scripts were taken from the following:

https://github.com/Chipdelmal/MoNeT_DA/blob/main/STP/RBC/STP_clsTrain.py

# Authors

Elijah Bartolome, Héctor M. Sánchez C.
