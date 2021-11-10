
#%%
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
WOP_var = normalize["WOP"]
CPT_var = normalize['CPT']

#%%
## function for lin reg
def linregression(indep_train, indep_test, dep_train, dep_test):
    """Returns the r2 and rmse values."""
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
results

# %%
###############################################################################
# take out individual variables to see if it helps (doesn't lol)
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

results

#%%
###############################################################################
# K Fold
###############################################################################
kfold_results = pd.DataFrame(columns=["WOP_r2", 'CPT_r2', "WOP_rmse", "CPT_rmse"])
k = 10
kf_WOP = KFold(n_splits=k)
#switch train data into numpy arrays 
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

kfold_results
# %%
