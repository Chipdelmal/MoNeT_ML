#%%
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datascience import *

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

##############################################
# load under numpy
##############################################
DF = Table().read_table(expPath)

# %%
sns.scatterplot(data=DATA, x='i_ren', y='WOP', hue='i_res')

# %%
# %%
DF.scatter('i_res', 'WOP')

#
##############################################
# preliminary plotting
##############################################
independent_vars = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']
for column in independent_vars:
    DF.scatter(column, 'WOP')
    DF.scatter(column, 'CPT')


# %%
###############################################################################
# calculate R values for each variable in relation to WOP and CPT
###############################################################################
def standard_units(arr):
    return (arr - np.mean(arr)) / np.std(arr)
def r_value(x, y):
    return np.mean(standard_units(DF.column(x)) * standard_units(DF.column(y)))

independent_vars = ['i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf']
WOP_r = []
CPT_r = []
for var in independent_vars:
    CPT_r.append(r_value(var, "CPT"))
    WOP_r.append(r_value(var, "WOP"))

r_value_table = Table().with_columns("independent var", independent_vars,'WOP r', WOP_r, 'CPT r', CPT_r)
r_value_table

# %%
###############################################################################
# get_dummies but numpy
###############################################################################
# for i_sex, 1 is male, 2 is males and not pregnant females, 3 is males and pregnant females
clean_DATA = DF.select('i_sex', 'i_ren', 'i_res', 'i_gsv', "i_fch", 'i_fcb', 'i_fcr' ,'i_hrf', "WOP", 'CPT')
clean_DATA = clean_DATA.with_columns(
    'i_sex_1', 1 * (clean_DATA.column('i_sex') == 1),
    'i_sex_2', 1 * (clean_DATA.column('i_sex') == 2),
    'i_sex_3', 1 * (clean_DATA.column('i_sex') == 3)).drop('i_sex')
clean_DATA.show(10)
# %%
###############################################################################
# split dataset
###############################################################################
eighty_percent = round(clean_DATA.num_rows * .8)
shuffled_DATA = clean_DATA.shuffle()
training = shuffled_DATA.take(np.arange(eighty_percent))
## DO NOT TOUCH TESTING TIL END
testing = shuffled_DATA.take(np.arange(eighty_percent, clean_DATA.num_rows))
# %%
