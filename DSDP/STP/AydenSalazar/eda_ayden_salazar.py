# %%
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

GDRIVE = 'LDR'
FILE_NAME = 'SCA_HLT_50Q_10T.csv'
BASE_PATH = '/Users/asalazar/Desktop/mosquito/'
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
###############################################################################
# Exploration plots
###############################################################################
# Basic scatter plots ---------------------------------------------------------
sns.scatterplot(data=df, x='i_ren', y='WOP', hue='i_res')
# Experiments distributions ---------------------------------------------------
sns.jointplot(data=DATA, x='i_fch', y='i_res')
# Heatmap ---------------------------------------------------------------------
dxy = 1000
(xAx, yAx) = (df['i_ren'], df['i_res'])
xi = np.linspace(min(xAx), max(xAx), dxy)
yi = np.linspace(min(yAx), max(yAx), dxy)
zi = griddata(
    (df['i_ren'], df['i_res']), df['CPT'], 
    (xi[None, :], yi[:, None]), 
    method='linear'
)
(fig, ax) = plt.subplots(figsize=(10, 8))
ax.contourf(xi, yi, zi, linewidths=.5, alpha=1, cmap='Blues')
# %%

# linear relationship between i_res (Size of the weekly releases) and WOP
# WOP => Window of Protection
sns.scatterplot(data=df, x='i_res', y='WOP', hue='i_res')

# %%
