# São Tomé and Príncipe

## Datasets

* `SCA_HLT_50Q_10T.csv`
* `CLS_HLT_50Q_10T.csv`
* `REG_HLT_50Q_10T.csv`

`CLS` is used for training classifiers while `REG` and `SCA` are used for training regressors.

## Python Scripts Nomenclature

`model` represents which model used to execute the task on the datasets (e.g. using an Extra Trees model)

* `STP_Train_model.py`    
    * Replace `model` with: 
        * `b`: Bagging Classifier/Regressor
        * `et`: Extra Trees
        * `gbt`: Gradient Boosted Trees
        * `rf`: Random Forest
        * `s`: Stacking Classifier/Regressor
        * `v`: Voting Classifier/Regressor

## Running Classifiers/Regressors

To run classifiers/regressors on a certain dataset and metric, 

```bash
./STP_Train.sh $dataset $metric $path
```

where `$dataset` is the three-letter abreviation of the dataset you're using, `$metric` is the metric abbreviation of your choice in a string, and `$path` is where the dataset csv files are and also where the output result files are saved to e.g. `./STP_Train.sh 'STP' 'POE' '/input_output'` to train a model on the `STP` dataset fitted on `POE` metric and the datasets are taken from the `/input_output` folder which is also where the result files are saved.

The `STP_Train.sh` file uses the given arguments on default models.

If you want to use a specfic model, you can alter the `STP_Train.sh` file or run a specific python script as follows:

```bash
./STP_Train_model.py $dataset $metric $path
```
where the arguments are the same as above and the `model` in the python script name is replaced with one of the model abbreviations from the Python Scripts Nomenclature section.

## Result Files

The above python scripts generate a report `.txt` file that shows meta-information such as the features/input variables and summary statistics of the model used such as a R2 score for the regressors. `.joblib` files are also generated. `.png` files showing a confusion matrix of the model results are also generated when using the `CLS` dataset.

## Result Files Nomenclature

`set` represents which dataset was used to train the models on.

`model` represents which model used to execute the task on the datasets (e.g. using an Extra Trees model)

`metric` represents the outcome variable the model was tasked to fit on.

`type` is either `.txt`, `.joblib`, or `.png`.

* `set_model_metric.txt`
    * Replace `set` with: 
        * `SCA`
        * `CLS`
        * `REG`
    * Replace `model` with: 
        * `B`: Bagging Classifier/Regressor
        * `ET`: Extra Trees
        * `GBT`: Gradient Boosted Trees
        * `RF`: Random Forest
        * `S`: Stacking Classifier/Regressor
        * `V`: Voting Classifier/Regressor
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
