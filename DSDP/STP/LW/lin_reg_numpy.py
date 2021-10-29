
#%%
from os import path
from posixpath import normcase
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from scipy.sparse.construct import rand
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from seaborn.utils import _normalize_kwargs
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing

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

# %%
###############################################################################
# Basic lin reg using built in sklearn function
###############################################################################
# selected = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', 'WOP', "CPT"]]
# cleanedDATA = clean_DATA = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', "WOP", 'CPT']]
## need to figure out how to clean the i_sex column
# getdummiesish = {
#     'i_sex_1': 1 * (clean_DATA.column('i_sex') == 1),
#     'i_sex_2': 1 * (clean_DATA.column('i_sex') == 2),
#     'i_sex_3': 1 * (clean_DATA.column('i_sex') == 3)}
# clean_DATA.head()
# cleanedDATA.loc[df['i_sex']]

#%%
## Old values from the first iteration of my program 
old_WOP_r2 = 0.7442248508814753
old_CPT_r2 = 0.7400280214376855
old_WOP_rmse = 350.1347831308501
old_CPT_rmse = 0.16631627860089154

# %%
###############################################################################
# clean dataset (one hot encoding, noramlize)
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
# %%
###############################################################################
# split dataset
###############################################################################
x_train, x_test, WOP_train, WOP_test = train_test_split(independent_vars, WOP_var, test_size=0.2, random_state=50)
z_train, z_test, CPT_train, CPT_test = train_test_split(independent_vars, CPT_var, test_size=0.2, random_state=50)

###############################################################################
# apply linear regression using sklearn's LinearRegression class
###############################################################################
LR_WOP = LinearRegression()
LR_WOP.fit(x_train, WOP_train)
WOP_predict = LR_WOP.predict(x_test)

LR_CPT = LinearRegression()
LR_CPT.fit(z_train, CPT_train)
CPT_predict = LR_CPT.predict(z_test)

# %%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

WOP_r2 = r2_score(WOP_test, WOP_predict)
CPT_r2 = r2_score(CPT_test, CPT_predict)
WOP_rmse = np.sqrt(mean_squared_error(WOP_test, WOP_predict))
CPT_rmse = np.sqrt(mean_squared_error(CPT_test, CPT_predict))

#%%
results = pd.DataFrame(data={'removed':['none'], 'WOPr2':[WOP_r2], 'CPTr2':[CPT_r2], 'WOPrmse':[WOP_rmse], 'CPTrmse':[CPT_rmse]})
results

#%%
x_train.drop(columns=['i_ren'])
# %%
###############################################################################
# take out individual variables to see if it helps
###############################################################################
indep_var_names = ['i_ren', 'i_res', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrf']
for var in indep_var_names:
    # WOP prediction
    wop_var = x_train.drop(columns=[var])
    wop_test = x_test.drop(columns=[var])
    linreg_wop = LinearRegression()
    linreg_wop.fit(wop_var, WOP_train)
    predicted_wop = linreg_wop.predict(wop_test)
    wop_r2 = r2_score(WOP_test, predicted_wop)
    wop_rmse = np.sqrt(mean_squared_error(WOP_test, predicted_wop))
    # CPT prediction 
    cpt_var = z_train.drop(columns=[var])
    cpt_test = z_test.drop(columns=[var])
    linreg_cpt = LinearRegression()
    linreg_cpt.fit(cpt_var, CPT_train)
    predicted_cpt = linreg_cpt.predict(cpt_test)
    cpt_r2 = r2_score(CPT_test, predicted_cpt)
    cpt_rmse = np.sqrt(mean_squared_error(CPT_test, predicted_cpt))
    df = pd.DataFrame(data={'removed':[var], 'WOPr2':[wop_r2], 'CPTr2':[cpt_r2], 'WOPrmse':[wop_rmse], 'CPTrmse':[cpt_rmse]})
    results = results.append(df)

i_sex_vars = ['i_sex_1', 'i_sex_2', 'i_sex_3']

results
#%%
## show results 
print("the WOP r2 is: " + str(WOP_r2))
print("the CPT r2 is: " + str(CPT_r2))
print("the WOP rmse is: " +  str(WOP_rmse))
print ("the CPT rmse is: " + str(CPT_rmse))
# %%

# writing a function to make my life easier 
