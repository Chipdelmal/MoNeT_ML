
#%%
from collections import defaultdict
from os import path
from posixpath import normcase
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from scipy.sparse.construct import rand, random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from seaborn.utils import _normalize_kwargs
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

GDRIVE = 'LDR'
FILE_NAME = 'SCA_HLT_50Q_10T.csv'
BASE_PATH = '/Users/lillianweng/Desktop/DSDP/mosquito_raw_data/'
###############################################################################
# Load dataset
###############################################################################
expPath = path.join(BASE_PATH, GDRIVE, FILE_NAME)
DATA = pd.read_csv(expPath)
print('* Dataset Path: {}'.format(expPath))
print('* Dataset Dimensions: {}'.format(DATA.shape))
DATA.head()

###############################################################################
# Filter to "center" parameters
###############################################################################
fltr = (
    (DATA['i_grp'] == 0)    &
    (DATA['i_sex'] == 1)    &
    np.isclose(DATA['i_fch'], 0.175)    &
    np.isclose(DATA['i_fcb'], 0.117)    &
    np.isclose(DATA['i_fcr'], 0)        &
    np.isclose(DATA['i_hrm'], 1.0)      &
    np.isclose(DATA['i_hrf'], 0.956)    &
    np.isclose(DATA['i_rsg'], 0.079)    &
    np.isclose(DATA['i_gsv'], 1.e-02) 
)
df = DATA[fltr]
df.head()

# %%
# linear relationship between i_res (Size of the weekly releases) and WOP
# WOP => Window of Protection
sns.scatterplot(data=df, x='i_res', y='WOP', hue='i_res')

# %%
###############################################################################
# preliminary plotting
###############################################################################
# independent_vars = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']
# sns.scatterplot(data=DATA, x='i_sex', y="WOP", hue='i_sex')
# sns.scatterplot(data=DATA, x='i_res', y="CPT", hue='i_sex')

#%%
## Old values from the first iteration of my program 
old_WOP_r2 = 0.7442248508814753
old_CPT_r2 = 0.7400280214376855
old_WOP_rmse = 350.1347831308501
old_CPT_rmse = 0.16631627860089154

# %%
###############################################################################
# clean dataset (one hot encoding, normalize)
###############################################################################
necessaryVars = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', "WOP", 'CPT']]
oneHotEncoding = pd.get_dummies(necessaryVars['i_sex'])
necessaryVars = necessaryVars.drop('i_sex', axis = 1)
necessaryVars = necessaryVars.join(oneHotEncoding)
cleaned = necessaryVars.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
# normalize = preprocessing.Normalizer() 
normalize = (cleaned - cleaned.mean()) / cleaned.std()
independent_vars = normalize.drop(columns=['WOP', 'CPT'])
independent_vars = independent_vars.reindex(sorted(independent_vars.columns), axis=1)
WOP_var = normalize["WOP"]
CPT_var = normalize['CPT']

#%%
## function for lin reg
def linregression(indep_train, indep_test, dep_train, dep_test):
    """Takes in 4 dataframes and trains the linear regression model 
    based on indep_train and dep_train, testing its accuracy with 
    indep_test and dep_test. 
    Returns the r2 and rmse values as a tuple."""
    LR = LinearRegression()
    LR.fit(indep_train, dep_train)
    predicted = LR.predict(indep_test)
    r2 = r2_score(dep_test, predicted)
    rmse = np.sqrt(mean_squared_error(dep_test, predicted))
    return r2, rmse

# %%
###############################################################################
# split dataset
###############################################################################
x_train, x_test, WOP_train, WOP_test = train_test_split(independent_vars, WOP_var, test_size=0.2, random_state=50)
z_train, z_test, CPT_train, CPT_test = train_test_split(independent_vars, CPT_var, test_size=0.2, random_state=50)

###############################################################################
# apply linear regression using sklearn's LinearRegression class
###############################################################################
WOP_r2, WOP_rmse = linregression(x_train, x_test, WOP_train, WOP_test)
CPT_r2, CPT_rmse = linregression(z_train, z_test, CPT_train, CPT_test)

## show results 
print("the WOP r2 is: " + str(WOP_r2))
print("the CPT r2 is: " + str(CPT_r2))
print("the WOP rmse is: " +  str(WOP_rmse))
print ("the CPT rmse is: " + str(CPT_rmse))
#%%
results = pd.DataFrame(data={'removed':['none'], 'WOPr2':[WOP_r2], 'CPTr2':[CPT_r2], 'WOPrmse':[WOP_rmse], 'CPTrmse':[CPT_rmse]})

# %%
###############################################################################
# take out individual variables to see if it helps (spoiler alert: doesn't help)
###############################################################################
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

i_sex_vars = ['i_sex_1', 'i_sex_2', 'i_sex_3']
wop_var = x_train.drop(columns=i_sex_vars)
wop_test = x_test.drop(columns=i_sex_vars)
wop_r2, wop_rmse = linregression(wop_var, wop_test, WOP_train, WOP_test)

 # CPT prediction 
cpt_var = z_train.drop(columns=i_sex_vars)
cpt_test = z_test.drop(columns=i_sex_vars)
cpt_r2, cpt_rmse = linregression(cpt_var, cpt_test, CPT_train, CPT_test)
df = pd.DataFrame(data={'removed':['i_sex'], 'WOPr2':[wop_r2], 'CPTr2':[cpt_r2], 'WOPrmse':[wop_rmse], 'CPTrmse':[cpt_rmse]})
results = results.append(df)

#%%
## display dataframe after systmatically removing one independent variable 
results.style.set_caption("Results After Removing Each Variable")

#%%
###############################################################################
# K Fold
###############################################################################
kfold_results = pd.DataFrame(columns=["WOP_r2", 'CPT_r2', "WOP_rmse", "CPT_rmse"])
k = 10
kf_WOP = KFold(n_splits=k)
# switch train data into numpy arrays 
wop_x = x_train.to_numpy()
wop_y = WOP_train.to_numpy()
cpt_x = z_train.to_numpy()
cpt_y = CPT_train.to_numpy()
for train_index, test_index in kf_WOP.split(x_train):
    wop_x_train, wop_x_test = wop_x[train_index], wop_x[test_index]
    wop_y_train, wop_y_test = wop_y[train_index], wop_y[test_index]
    cpt_x_train, cpt_x_test = cpt_x[train_index], cpt_x[test_index]
    cpt_y_train, cpt_y_test = cpt_y[train_index], cpt_y[test_index]
    wop_r2, wop_rmse = linregression(wop_x_train, wop_x_test, wop_y_train, wop_y_test)
    cpt_r2, cpt_rmse = linregression(cpt_x_train, cpt_x_test, cpt_y_train, cpt_y_test)
    df = pd.DataFrame({"WOP_r2":[wop_r2], 'CPT_r2':[cpt_r2], "WOP_rmse":[wop_rmse], "CPT_rmse":[cpt_rmse]})
    kfold_results = pd.concat([kfold_results, df], ignore_index = True, axis=0)

#%%
## display dataframe for K Fold results 
kfold_results.style.set_caption("K Fold Results")
# %%
###############################################################################
# Final Model
###############################################################################
final_wop_alg = LinearRegression()
final_wop_alg.fit(x_train, WOP_train)
predicted_wop = final_wop_alg.predict(x_test) # numpy array 
final_cpt_alg = LinearRegression()
final_cpt_alg.fit(z_train, CPT_train)
predicted_cpt= final_cpt_alg.predict(z_test) # numpy array 

#%%
## table with actual vs predicted WOP and CPT values 
plot_results_df = pd.DataFrame()
plot_results_df['wop_actual'] = WOP_test.to_list()
plot_results_df['wop_predict'] = predicted_wop
plot_results_df['cpt_actual'] = CPT_test.to_list()
plot_results_df['cpt_predict'] = predicted_cpt

#%%
## plot WOP predicted vs actual
sns.scatterplot(data=plot_results_df, x='wop_predict', y='wop_actual').set(title="WOP Predicted vs. Actual")

#%%
## plot CPT predicted vs actual
sns.scatterplot(data=plot_results_df, x='cpt_predict', y='cpt_actual').set(title="CPT Predicted vs. Actual")

#%%
## table of the coefficients determined by sci-kit learn's Linear Regression 
coefficients = pd.DataFrame(final_wop_alg.coef_, x_test.columns, columns=["WOP Coefficients"])
coefficients["CPT Coefficients"] = final_cpt_alg.coef_
coefficients.style.set_caption("Coefficients")
#%% 
adjustment_results = pd.DataFrame(data={'WOP adjustment':['no change'], 'WOPr2':[WOP_r2], 'WOPrmse':[WOP_rmse], 'CPT adjustment':['no change'], 'CPTr2':[CPT_r2], 'CPTrmse':[CPT_rmse]})

def r2_and_rmse(actual, predicted):
    return r2_score(actual, predicted), np.sqrt(mean_squared_error(actual, predicted))

#%%
## making predictions better match actual
adjust_results_df = plot_results_df.copy(deep=True)
## WOP predicted that are greater than 1 are changed to 1
adjust_results_df.loc[(adjust_results_df.wop_predict >= 1), "wop_predict"] = 1
## CPT predicted that are less than -1 are changed to -1
adjust_results_df.loc[(adjust_results_df.cpt_predict <= -1), "cpt_predict"] = -1
wopr2, woprmse = r2_and_rmse(adjust_results_df["wop_predict"], plot_results_df["wop_actual"])
cptr2, cptrmse = r2_and_rmse(adjust_results_df["cpt_predict"], plot_results_df["cpt_actual"])
dict = {'WOP adjustment':'if > 1, change to 1', 'WOPr2':wopr2, 'WOPrmse':woprmse, 'CPT adjustment':'if < -1, change to -1', 'CPTr2':cptr2, 'CPTrmse':cptrmse}
adjustment_results = adjustment_results.append(dict, ignore_index=True)
adjustment_results

#%%
## plot WOP after adjusting 
sns.scatterplot(data=adjust_results_df, x='wop_predict', y='wop_actual').set(title="WOP Adjusted Predicted vs. Actual")
#%%
## plot CPT after adjusting 
sns.scatterplot(data=adjust_results_df, x='cpt_predict', y='cpt_actual').set(title="CPT Adjusted Predicted vs. Actual")

#%%
## exponentialize WOP
adjust_results_df["wop_predict"] = np.exp(adjust_results_df["wop_predict"])
## log CPT
adjust_results_df["cpt_predict"] = np.log(adjust_results_df["cpt_predict"] + 2) ## this makes r2 worse (0.76 --> 0.39)
wopr2, woprmse = r2_and_rmse(adjust_results_df["wop_predict"], plot_results_df["wop_actual"])
cptr2, cptrmse = r2_and_rmse(adjust_results_df["cpt_predict"], plot_results_df["cpt_actual"])
dict = {'WOP adjustment':'exponentialize', 'WOPr2':wopr2, 'WOPrmse':woprmse, 'CPT adjustment':'log', 'CPTr2':cptr2, 'CPTrmse':cptrmse}
adjustment_results = adjustment_results.append(dict, ignore_index=True)
adjustment_results
#%%
## plot exponentialized WOp
sns.scatterplot(data=adjust_results_df, x='wop_predict', y='wop_actual').set(title="WOP Predicted vs. Actual")
#%% 
## plot logged CPT 
sns.scatterplot(data=adjust_results_df, x='cpt_predict', y='cpt_actual').set(title="CPT Predicted vs. Actual")

#%%
plot_results_df['adjusted_wop_predict'] = adjust_results_df["wop_predict"]
plot_results_df["adjusted_cpt_predict"] = adjust_results_df["cpt_predict"]
plot_results_df.style.set_caption('Predictions vs Actual of Final Algorithm')
plot_results_df