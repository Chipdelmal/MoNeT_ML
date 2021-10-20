
#%%
from os import path
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from scipy.sparse.construct import rand
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

# %%
# linear relationship between i_res (Size of the weekly releases) and WOP
# WOP => Window of Protection
sns.scatterplot(data=df, x='i_res', y='WOP', hue='i_res')

# %%
###############################################################################
# preliminary plotting
###############################################################################
independent_vars = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']
sns.scatterplot(data=DATA, x='i_sex', y="WOP", hue='i_sex')
sns.scatterplot(data=DATA, x='i_res', y="CPT", hue='i_sex')

# %%
###############################################################################
# Basic lin reg using built in sklearn function
###############################################################################
from sklearn import linear_model
# selected = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', 'WOP', "CPT"]]

# %%
# cleanedDATA = clean_DATA = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', "WOP", 'CPT']]
## need to figure out how to clean the i_sex column
# getdummiesish = {
#     'i_sex_1': 1 * (clean_DATA.column('i_sex') == 1),
#     'i_sex_2': 1 * (clean_DATA.column('i_sex') == 2),
#     'i_sex_3': 1 * (clean_DATA.column('i_sex') == 3)}
# clean_DATA.head()
# cleanedDATA.loc[df['i_sex']]



# %%
cleanedDATA = clean_DATA = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', "WOP", 'CPT']]
independent_vars = cleanedDATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']]
WOP_var = cleanedDATA["WOP"]
CPT_var = cleanedDATA['CPT']
# %%
###############################################################################
# split dataset
###############################################################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train, x_test, WOP_train, WOP_test = train_test_split(cleanedDATA, WOP_var, test_size=0.2, random_state=50)
z_train, z_test, CPT_train, CPT_test = train_test_split(cleanedDATA, CPT_var, test_size=0.2, random_state=50)


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
## show results 
print("the WOP r2 is: " + str(WOP_r2))
print("the CPT r2 is: " + str(CPT_r2))
print("the WOP rmse is: " +  str(WOP_rmse))
print ("the CPT rmse is: " + str(CPT_rmse))
# %%
