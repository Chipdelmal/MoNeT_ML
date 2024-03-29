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
from itertools import combinations

# comb = combinations(df.columns, 2)
# print(len(list(comb)))
# i = 1
# for x, y in comb:
#     plt.subplot(2,2,i)
#     print(x, y)
#     sns.scatterplot(data=df, x=x, y=y)
#     i += 1


# %%
interesting_features = ['i_ren', 'i_res', 'TTI', 'TTO', 'CPT', 'WOP']
comb = combinations(interesting_features, 2)
i = 1
for x, y in comb:
    if i==7:
        break
    plt.subplot(2,3,i)
    print(x, y)
    sns.scatterplot(data=df, x=x, y=y)
    i += 1
# %%
# using Keras to create neural network
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
# male:1 male and non-pregnant females: 2 males and pregnant females: 3
# Do normalize your data! 
X = df.iloc[:, 0:12]
y = df.iloc[:, [17]] # 17 for CPT
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu')) #Densely connected: Each neuron is connected all the neurons in the next layer
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=100)

print('RMSE on training---')
y_pred_train=model.predict(X_train)
print(mean_squared_error(y_train,y_pred_train))

print('RMSE  on test---')
y_pred_test=model.predict(X_test)
print(mean_squared_error(y_test,y_pred_test))

# %%
y_train,y_pred_train
# %%
