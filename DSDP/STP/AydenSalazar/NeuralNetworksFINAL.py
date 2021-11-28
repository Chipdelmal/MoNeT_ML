# %%
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

df_list = []
for gdrive in ['LDR', 'SDR']:
    for file_type in ['SCA', 'CLS']:
        GDRIVE = gdrive
        FILE_NAME = '{}_HLT_50Q_10T.csv'.format(file_type)
        BASE_PATH = '/Users/asalazar/Desktop/mosquito/'

        # Load dataset
        expPath = path.join(BASE_PATH, GDRIVE, FILE_NAME)
        DATA = pd.read_csv(expPath)
        print('* Dataset Path: {}'.format(expPath))
        print('* Dataset Dimensions: {}'.format(DATA.shape))
        df = DATA
        df_list.append(df)

# %%
# Convert sex column into one-hot encoded columns
for i, df in enumerate(df_list):
    if 'i_sex' in df.columns:

        # assign one-hots
        oh = pd.get_dummies(df['i_sex'], prefix='i_sex')
        new_df = pd.concat([df, oh], axis=1).drop(['i_sex'], axis=1)

        df_list[i] = new_df

df_ldr_sca = df_list[0]
df_ldr_cls = df_list[1]
        
df_sdr_sca = df_list[2]
df_sdr_cls = df_list[3]

df_sdr_sca
# %%
# Let's hone in on the df_sdr_sca dataset. First, let's identify a X, y set with target CPT


X = df_sdr_sca.loc[:, ['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrm', 'i_hrf',	'i_grp', 'i_mig', 'i_sex_1', 'i_sex_2', 'i_sex_3']] # features i_ren => i_sex_3
#X = pd.concat([X, df_sdr_sca[['i_sex_1', 'i_sex_2', 'i_sex_3']]])
y = df_sdr_sca.iloc[:, [16]] # index 16 is CPT

# leave X a dataframe (function will convert it to array)
y = np.array(y)

print(X.shape)
print(y.shape)
# %%
# minibatch gradient descent for CPT

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc=preprocessing.MinMaxScaler()
X_train = sc.fit_transform(X_train)

sc=preprocessing.MinMaxScaler()
X_test = sc.fit_transform(X_test)


n_train = 500

# define model

num_dim = X_train.shape[1]
print("Num dim:", num_dim)
model = Sequential()
model.add(Dense(30, input_dim=num_dim, activation='relu')) 
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) # optimizer adam works

# fit model
print("LEN of X_TRAIN:", X_test.shape, "len of y_train:", y_test.shape)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=500) # works well with 500 batch size


# %%
y_pred = model.predict(X_test)
mean_squared_error(y_pred, y_test)  
# %%
plt.xlabel("Actual CPT")
plt.ylabel("Predicted CPT")
plt.title("Actual vs Predicted CPT using Batch-Trained Neural Network on All Features")
plt.scatter(y_test, y_pred, s=.1)
plt.plot(y_test, y_test, color = 'red', label="Actual = Predicted")
plt.legend()
#plt.axes().set_aspect(1.0/plt.axes().get_data_ratio(), adjustable='box')

# %%
# Let's hone in on the df_sdr_sca dataset. First, let's identify a X, y set with targe = window of protection


X = df_sdr_sca.loc[:, ['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrm', 'i_hrf',	'i_grp', 'i_mig', 'i_sex_1', 'i_sex_2', 'i_sex_3']] # features i_ren => i_sex_3
#X = pd.concat([X, df_sdr_sca[['i_sex_1', 'i_sex_2', 'i_sex_3']]])
y = df_sdr_sca.iloc[:, [13]] # index 13 is WOP

# # leave X a dataframe (function will convert it to array)
y = np.array(y)

print(X.shape)
print(y.shape)
# %%
df_sdr_sca.iloc[:, [13]]
# %%
# minibatch gradient descent for WOP

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc=preprocessing.MinMaxScaler()
X_train = sc.fit_transform(X_train)

sc=preprocessing.MinMaxScaler()
X_test = sc.fit_transform(X_test)


n_train = 500

# define model

num_dim = X_train.shape[1]
print("Num dim:", num_dim)
model = Sequential()
model.add(Dense(30, input_dim=num_dim, activation='relu')) 
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

# compile model
opt = SGD(lr=0.01, momentum=0.9)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 
model.compile(loss='huber', optimizer=opt, metrics=['accuracy']) # huber loss works good 

# fit model
print("LEN of X_TRAIN:", X_test.shape, "len of y_train:", y_test.shape)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100) # works well with 500 batch size

# %%
y_pred = model.predict(X_test)
mean_squared_error(y_pred, y_test)  
# %%
plt.xlabel("Actual WOP")
plt.ylabel("Predicted WOP")
plt.title("Actual vs Predicted WOP using Batch-Trained Neural Network on All Features")
plt.scatter(y_test, y_pred, s=.1)
plt.plot(y_test, y_test, color = 'red', label="Actual = Predicted")
plt.legend()
#plt.set_aspect('equal')

# %%
