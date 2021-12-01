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

#%%
###############################################################################
# Clean dataset (one hot encoding, normalize)
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

# %%
###############################################################################
# Split dataset (same seed for both WOP and CPT)
###############################################################################
x_train, x_test, WOP_train, WOP_test = train_test_split(independent_vars, WOP_var, test_size=0.2, random_state=50)
z_train, z_test, CPT_train, CPT_test = train_test_split(independent_vars, CPT_var, test_size=0.2, random_state=50)


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
def predict(list):
    """ Takes in a list of integers in the order: ['i_sex', 'i_ren', 'i_res', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrf'].
        Normalizes the data before running it thorugh the algorithm.
        Returns predicted WOP and CPT value in original units. """
    
    ## make list input into a dataframe
    input_df = pd.DataFrame(columns = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf'])
    input_df.loc[0] = list
    ## one hot encoding
    oneHotEncoding = pd.get_dummies(input_df['i_sex'])
    input_df = input_df.drop('i_sex', axis = 1)
    input_df = input_df.join(oneHotEncoding)
    input_df = input_df.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
    cleaned_dropped = cleaned.drop(columns=["CPT", "WOP"])
     ## normalize data using cleaned table from above 
    input_df = (input_df - cleaned_dropped.mean()) / cleaned_dropped.std()
    input_df = input_df.fillna(x_test["i_sex_1"].to_list()[0])
    input_df = input_df.reindex(sorted(input_df.columns), axis=1)
    # predict values 
    predict_wop = final_wop_alg.predict(input_df)
    if predict_wop > 1:
        predict_wop = 1
    predict_wop =  np.exp(predict_wop)
    predict_cpt = final_cpt_alg.predict(input_df)
    if predict_cpt < -1:
        predict_cpt = -1
    ## turn back into origianl units 
    return predict_wop * cleaned["WOP"].std() + cleaned["WOP"].mean(), predict_cpt * cleaned["CPT"].std() + cleaned["CPT"].mean()


#%%
###############################################################################
## testing the final function
###############################################################################
# test_list = independent_vars[]
testing_df = DATA[['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']]
test_wop, test_cpt = predict(testing_df.loc[437275].to_list())
print("wop: " + str(test_wop) + " \ncpt: " + str(test_cpt))

# x_test.head(1)
# %%
x_test

## make list input into a dataframe
list = testing_df.loc[437275].to_list()
input_df = pd.DataFrame(columns = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf'])
input_df.loc[0] = list
## one hot encoding
oneHotEncoding = pd.get_dummies(input_df['i_sex'])
input_df = input_df.drop('i_sex', axis = 1)
input_df = input_df.join(oneHotEncoding)
input_df = input_df.rename(columns={1:"i_sex_1", 2:"i_sex_2", 3:"i_sex_3"})
# # normalize data using cleaned table from above 
cleaned_dropped = cleaned.drop(columns=["CPT", "WOP"])
input_df = (input_df - cleaned_dropped.mean()) / cleaned_dropped.std()
input_df = input_df.fillna(x_test["i_sex_1"].to_list()[0])
input_df = input_df.reindex(sorted(input_df.columns), axis=1)
input_df
# input_df.equals(x_test.head(1))
# predict_wop = final_wop_alg.predict(input_df)
# predict_wop
# if predict_wop > 1:
#     predict_wop = 1
# predict_wop =  np.exp(predict_wop)
# predict_cpt = final_cpt_alg.predict(input_df).item(0)
# predict_cpt
# if predict_cpt < -1:
#     predict_cpt = -1
# ## turn back into origianl units 
# wop, cpt =  predict_wop * cleaned["WOP"].std() + cleaned["WOP"].mean(), predict_cpt * cleaned["CPT"].std() + cleaned["CPT"].mean()
# print(str(predict_wop) + str(predicted_cpt))
# print(str(wop) + str(cpt))
# %%
# %%
