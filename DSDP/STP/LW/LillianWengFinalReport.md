# Data Science Discovery Program: Using Linear Regression to Predict the Window of Protection and Cumulative Potential for Transmission for Genetically Modified Mosquitos
### Abstract
### Objective:
Train a machine learning model to predict the effectiveness of genetially modified mosquitoes in decreasing the frequency of mosquito-transmitted diseases like malaria and dengue feaver. Summary statistics were provided by UC Berkeley School of Public Health's Marshall Lab. More specifically, I chose linear regression to train 2 separate models for predicting the Window of Protection (WOP) and the CUmulative Potential for Transmission (CPT) using the LDR dataset provided by the lab. 
### Data Cleaning 
Since the "i_sex" column designating the gender of the released mosquitos was a categorical variable, I first converted it to a quantitative value using one-hot encoding then normalized the data.
```
necessaryVars = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', "WOP", 'CPT']]
oneHotEncoding = pd.get_dummies(necessaryVars['i_sex'])
necessaryVars = necessaryVars.drop('i_sex', axis = 1)
necessaryVars = necessaryVars.join(oneHotEncoding)
cleaned = necessaryVars.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
normalize = (cleaned - cleaned.mean()) / cleaned.std()
```
Using sci-kit learn's model selection package, I split the data into training and testing using an 80:20 ratio; I used the same random seed to ensure that both models are consistent.
```
x_train, x_test, WOP_train, WOP_test = train_test_split(independent_vars, WOP_var, test_size=0.2, random_state=50)
z_train, z_test, CPT_train, CPT_test = train_test_split(independent_vars, CPT_var, test_size=0.2, random_state=50)
```

### Exploration 
To determine the effectiveness of the models I test, I wrote the following function that returns the r2 score and root mean squared error (RMSE) calculated using sci-kit learn's metrics package.  
```
def linregression(indep_train, indep_test, dep_train, dep_test):
    """Takes in 4 dataframes and trains the linear regression model 
    based on indep_train and dep_train, testing its accuracy with 
    indep_test and dep_test. 
    Returns the r2 and rmse values as a tuple."""
    LR = LinearRegression()
    LR.fit(indep_train, dep_train)
    predicted = LR.predict(indep_test)
    # r2 = r2_score(dep_test, predicted)
    r2 = rSquared(dep_test, predicted)
    rmse = np.sqrt(mean_squared_error(dep_test, predicted))
    return r2, rmse
```
I started by systematically removing one of the independent variables i_sex, i_ren, i_res, i_gsv, i_fch, i_fcb, i_fcr, and i_hrf. If the r2 score or rmse dropped significantly with its removal, that indicated the variable is significant and should be kept. 

```
indep_var_names = ['i_ren', 'i_res', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrf']
for var in indep_var_names:
    # WOP prediction
    wop_var = x_train.drop(columns=[var])
    wop_test = x_test.drop(columns=[var])
    wop_r2, wop_rmse = linregression(wop_var, wop_test, WOP_train, WOP_test)

    # CPT prediction 
    cpt_var = z_train.drop(columns=[var])
    cpt_test = z_test.drop(columns=[var])
    cpt_r2, cpt_rmse = linregression(cpt_var, cpt_test, CPT_train, CPT_test)
    df = pd.DataFrame(data={'removed':[var], 'WOPr2':[wop_r2], 'CPTr2':[cpt_r2], 'WOPrmse':[wop_rmse], 'CPTrmse':[cpt_rmse]})
    results = results.append(df)
```

DISPLAY THE TABLE
From Figure 1, removing each variable decreased the r2 score and RMSE, and the most accurate model kept all seven independent variables.


To ascertain the results of the model, I performed a K Fold test using sci-kit learn's model selection package and k=10. 
```
kfold_results = pd.DataFrame(columns=["WOP_r2", 'CPT_r2', "WOP_rmse", "CPT_rmse"])
k = 10
f_WOP = KFold(n_splits=k)
for train_index, test_index in kf_WOP.split(x_train):
    wop_x_train, wop_x_test = wop_x[train_index], wop_x[test_index]
    wop_y_train, wop_y_test = wop_y[train_index], wop_y[test_index]
    cpt_x_train, cpt_x_test = cpt_x[train_index], cpt_x[test_index]
    cpt_y_train, cpt_y_test = cpt_y[train_index], cpt_y[test_index]
    wop_r2, wop_rmse = linregression(wop_x_train, wop_x_test, wop_y_train, wop_y_test)
    cpt_r2, cpt_rmse = linregression(cpt_x_train, cpt_x_test, cpt_y_train, cpt_y_test)
    df = pd.DataFrame({"WOP_r2":[wop_r2], 'CPT_r2':[cpt_r2], "WOP_rmse":[wop_rmse], "CPT_rmse":[cpt_rmse]})
    kfold_results = pd.concat([kfold_results, df], ignore_index = True, axis=0)
```
DISPLAY THE TABLE
As can be seen in Figure 2, the r2 scores and RMSE values did not fluctuate significantly, leading to the conclusion that the results are not random and can be trusted. 

Given this information, I decided to use the Linear Regression model as is without deleting any columns

### Tweaking Model 
The final model with all seven independent variables included resulted in the following 2 scatterplots: 
DISPLAY SCATTERPLOTS Figure 3

#### Tails
For the WOP values, there seems to be a tail at the far right end, so I tried adjusting the values so that any predicted WOP values greater than 1 would map to 1.0. 
```
adjust_results_df.loc[(adjust_results_df.wop_predict >= 1), "wop_predict"] = 1
```

For the CPT values, there seems to be a tail at the bottom left end, so I tried adjusting the values so that any predicted CPT values below -1 mapped to -1.0
```
adjust_results_df.loc[(adjust_results_df.cpt_predict <= -1), "cpt_predict"] = -1
```
The resulting scatterplots are shown below: DISPLAY SCATTERPLOTS Figure 4