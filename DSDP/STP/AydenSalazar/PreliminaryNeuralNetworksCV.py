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
def mse_k_fold_no_feat(k, d_array, mse, X, y, rand_state):
    '''This function will print the d (degree) value that gives the smallest MSE.
    It also returns a dictionary of average MSEs (from k-fold cross validation) across the specified d values
    '''
    
    # 1. Initialize K-Fold CV where k = k
    kf = KFold(n_splits=k, random_state=rand_state, shuffle=True)
    
    # 2. Save the MSEs of each split.
    mses = np.full((k,1),np.nan)
    # above, we're initializing an array where:
    # every row of mses corresponds to one of the folds
    # every column of mses corresponds to one of the possible d values in d_array
    # for example, mses[0,0] corresponds to the mean squared error for the first fold, using d = 1 (the first element of d_array)
    
    fold = 0 # initialize fold value|
    for train_i, val_i in kf.split(X): # this loop iterates through the K folds of our data
        # 2.1 Separate X and Y_obs array into testing and validation sets
        X_fold_train = X.iloc[train_i, :]
        #print(train_i)
        #print("Y SHAPE", y.shape)
        y_fold_train = y[train_i]
        X_fold_val = X.iloc[val_i, :]
        y_fold_val = y[val_i]
        
        # Number of times to train new neural network model per fold
        for i in [1]: # this loop iterates through d values to produce a new model for each d value
            #d = list(d)
            print("LEN of X_FOLD_TRAIN:", X_fold_train.shape, "len of y_fold_train:", y_fold_train.shape)
            X_fold_train_temp = X_fold_train
            X_fold_val_temp = X_fold_val

            # normalize the data
            sc=preprocessing.MinMaxScaler()
            X_fold_train_temp = sc.fit_transform(X_fold_train_temp)

            sc=preprocessing.MinMaxScaler()
            X_fold_val_temp  = sc.fit_transform(X_fold_val_temp)

            # Initialize the model
            model = Sequential()

            # Initialize layers
            num_dim = X_fold_train_temp.shape[1]
            print("Num dim:", num_dim)

            model.add(Dense(30, input_dim=num_dim, activation='relu')) 
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X_fold_train_temp, y_fold_train, epochs=1)

            y_pred=model.predict(X_fold_val_temp)
            
            # Save each mse between y_pred and y_fold_val at their respective fold and d value
            mses[fold, i - 1] = mean_squared_error(y_pred, y_fold_val)
        
        fold += 1 # augment the fold count
    
    # 3. Now, find the average of the MSEs for each d value. 
    # Your result should be a dictionary with the same number of elements as d_array.
    # The keys of the dictionary should the the d-values. 
    # Each dictionary value is the average MSE across all k folds associated with its respective key, i.e., the d-value.
    # average_mses = {} # initialize an empty dictionary
    # for i in range(len(d_array)):
    #     average_mses[d_array[i]] = np.mean(mses[:, i]) # For each pass through the loop, add a dictionary entry in which the key is the d-value and the value is the average MSE
    
    # # 4. Find the index of the minimum average MSE
    # min_mse_index = min(average_mses, key=average_mses.get)
    
    # print("Minimum MSE Parameters:", min_mse_index, '\n',
    #       "MSE of {} Parameters:".format(min_mse_index), average_mses[min_mse_index])
    print("AVERAGE MSE (ACROSS ALL FOLDS: ", np.mean(mses))
    return mses


    
# %%
def mse_k_fold_lr(k, d_array, mse, X, y, rand_state):
    '''This function will print the d (degree) value that gives the smallest MSE.
    It also returns a dictionary of average MSEs (from k-fold cross validation) across the specified d values
    '''
    
    # 1. Initialize K-Fold CV where k = k
    kf = KFold(n_splits=k, random_state=rand_state, shuffle=True)
    
    # 2. Save the MSEs of each split.
    mses = np.full((k,len(d_array)),np.nan)
    # above, we're initializing an array where:
    # every row of mses corresponds to one of the folds
    # every column of mses corresponds to one of the possible d values in d_array
    # for example, mses[0,0] corresponds to the mean squared error for the first fold, using d = 1 (the first element of d_array)
    
    fold = 0 # initialize fold value
    for train_i, val_i in kf.split(X): # this loop iterates through the K folds of our data
        # 2.1 Separate X and Y_obs array into testing and validation sets
        X_fold_train = X.loc[train_i, :]
        y_fold_train = y[train_i]
        X_fold_val = X.loc[val_i, :]
        y_fold_val = y[val_i]
        
        # Each d value will receive its own neural network model.
        for i, d in enumerate(d_array): # this loop iterates through d values to produce a new model for each d value
            d = list(d)
            print("LEN of X_FOLD_TRAIN:", X_fold_train.shape, "len of y_fold_train:", y_fold_train.shape)
            X_fold_train_temp = X_fold_train.loc[:, d]
            X_fold_val_temp = X_fold_val.loc[:, d]

            # normalize the data
            sc=preprocessing.StandardScaler()
            X_fold_train_temp = sc.fit_transform(X_fold_train_temp)

            sc=preprocessing.StandardScaler()
            X_fold_val_temp  = sc.fit_transform(X_fold_val_temp)

            # Initialize the model
            model = Sequential()

            # Initialize layers
            num_dim = len(d)
            print("Num dim:", num_dim)

            model.add(Dense(30, input_dim=num_dim, activation='relu')) 
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X_fold_train_temp, y_fold_train, epochs=1)

            y_pred=model.predict(X_fold_val_temp)
            
            # Save each mse between y_pred and y_fold_val at their respective fold and d value
            mses[fold, i] = mean_squared_error(y_pred, y_fold_val)
        
        fold += 1 # augment the fold count
    
    # 3. Now, find the average of the MSEs for each d value. 
    # Your result should be a dictionary with the same number of elements as d_array.
    # The keys of the dictionary should the the d-values. 
    # Each dictionary value is the average MSE across all k folds associated with its respective key, i.e., the d-value.
    average_mses = {} # initialize an empty dictionary
    for i in range(len(d_array)):
        average_mses[d_array[i]] = np.mean(mses[:, i]) # For each pass through the loop, add a dictionary entry in which the key is the d-value and the value is the average MSE
    
    # 4. Find the index of the minimum average MSE
    min_mse_index = min(average_mses, key=average_mses.get)
    
    print("Minimum MSE Parameters:", min_mse_index, '\n',
          "MSE of {} Parameters:".format(min_mse_index), average_mses[min_mse_index])
    return average_mses
# %%
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

# find combos of interesting features
interesting_features = ['i_ren', 'i_res']
combos = powerset(interesting_features)
combos_list = list(combos)[1:]


combos_list

# %%
def fit_model(trainX, trainy, testX, testy, n_batch):
	
    # normalize the data
    sc=preprocessing.MinMaxScaler()
    trainX = sc.fit_transform(trainX)

    sc=preprocessing.MinMaxScaler()
    testX  = sc.fit_transform(testX)

    # Initialize the model
    model = Sequential()

    # Initialize layers
    num_dim = trainX.shape[1]
    print("Num dim:", num_dim)

    model.add(Dense(30, input_dim=num_dim, activation='relu')) 
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = SGD(lr=0.01, momentum=0.9)


    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, batch_size= n_batch)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title('batch='+str(n_batch), pad=-40)

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
plt.axes().set_aspect(1.0/plt.axes().get_data_ratio(), adjustable='box')

# %%
# Let's hone in on the df_sdr_sca dataset. First, let's identify a X, y set with targe = window of protection


X = df_sdr_sca.loc[:, ['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrm', 'i_hrf',	'i_grp', 'i_mig', 'i_sex_1', 'i_sex_2', 'i_sex_3']] # features i_ren => i_sex_3
#X = pd.concat([X, df_sdr_sca[['i_sex_1', 'i_sex_2', 'i_sex_3']]])
y = df_sdr_sca.iloc[:, [13]] # index 16 is CPT

# # leave X a dataframe (function will convert it to array)
y = np.array(y)

print(X.shape)
print(y.shape)

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
plt.xlabel("Actual WOP")
plt.ylabel("Predicted WOP")
plt.title("Actual vs Predicted WOP using Batch-Trained Neural Network on All Features")
plt.scatter(y_test, y_pred, s=.1)
plt.plot(y_test, y_test, color = 'red', label="Actual = Predicted")
plt.legend()
plt.set_aspect('equal')
# %%

# %%
