# Data Science Discovery Program: Using Linear Regression to Predict the Window of Protection and Cumulative Potential for Transmission for Genetically Modified Mosquitos
### Abstract
### Objective:
Train a machine learning model to predict the effectiveness of genetially modified mosquitoes in decreasing the frequency of mosquito-transmitted diseases like malaria and dengue feaver. Summary statistics were provided by UC Berkeley School of Public Health's Marshall Lab. More specifically, I chose linear regression to train 2 separate models for predicting the Window of Protection (WOP) and the CUmulative Potential for Transmission (CPT) using the LDR and SDR datasets provided by the lab. The analysis are based mainly off the SCA data provided.
### Data Cleaning 
Since the "i_sex" column designating the gender of the released mosquitos was a categorical variable, I first converted it to a quantitative value using one-hot encoding. I then normalized the data.
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
The two steps above were done do adjust both LDR and SDR data.

### Exploration 
To determine the effectiveness of the models I test, I used the r-squared adjusted value and the root mean squared error.

I began by systematically removing one of the independent variables (i_sex, i_ren, i_res, i_gsv, i_fch, i_fcb, i_fcr, and i_hrf). A significant drop in the r-squared adjusted or rmse values indicates that it should be kept in the final model.  

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

![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure1_remove_vars.jpg)

As seen in the tables, this procedure had a similar affect for both LDR and SDR datasets. Removing i_fch and i_fcb, and i_fcr decreased the r-squared adjusted value dramatically and increased the root mean squared errors for both WOP and CPT. The other variables had little affect on the r-squared adjusted and root mean squared error, but since the former decreased slightly and the latter increased slightly for each variable, I decided to retain all seven independent variables for my model.


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
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure2_kfold.jpg)

For both LDR and SDR datasets, the K-Fold results were similar for each of the ten iterations as expected.

Given this information, I decided to use the Linear Regression model as is without deleting any columns

### Tweaking The Model - LDR
The final model with all seven independent variables included resulted in the following scatterplots: 

![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure3_originalplots.jpg)

The WOP scatterplot has a tail at the far right end, so I adjusted the values so that any predicted WOP values greater than 1 would map to 1.0. 
```
adjust_results_df.loc[(adjust_results_df.wop_predict >= 1), "wop_predict"] = 1
```

The CPT scatterplot has a tail at the bottom left end, so I adjusted the values so that any predicted CPT values below -1 mapped to -1.0
```
adjust_results_df.loc[(adjust_results_df.cpt_predict <= -1), "cpt_predict"] = -1
```
The resulting scatterplots are shown below: 

![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure4_adj1.jpg)

The WOP plot looked exponential, and the CPT plot looked logorithmic, so I adjusted them respectively. 

![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure5_adj2.jpg)

However, neither procedure seemed beneficial. The WOP scatterplot now looked heteroskedastic, and the CPT scatterplot still looked logorithmic. These conclusions are also supported by the r-squared adjusted and root mean squared error values 

![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure6_adjtable.jpg)

For the final model, I decided to keep the first adjustment and drop the second one as per Dr. Sanchez's suggestions. 

### Tweaking The Model - SDR
The final model with all seven independent variables included resulted in the following 2 scatterplots: 

![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/LillianWeng/figures/figure7_originalplots.jpg)

