# %%
###############################################################################
# Getting LDR dataset
###############################################################################
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import r2_score

LDR_GDRIVE = 'dsdp_datasets/LDR'
FILE_NAME = 'SCA_HLT_50Q_10T.csv'
BASE_PATH = '/Users/joannayoo/Desktop'
# %%
###############################################################################
# Load dataset
###############################################################################
expPath = path.join(BASE_PATH, LDR_GDRIVE, FILE_NAME)
LDR_DATA = pd.read_csv(expPath)
print('* Dataset Path: {}'.format(expPath))
print('* Dataset Dimensions: {}'.format(LDR_DATA.shape))
LDR_DATA.head()
# %%
###############################################################################
# Data cleaning pipeline to get ready for model
###############################################################################
def data_stand_ohenc_pipeline(curr_df):
    #make a copy
    copy_df = curr_df.copy()
    #one hot encoding
    oh_enc = OneHotEncoder()
    oh_enc.fit(copy_df[['i_sex']])
    dummies = pd.DataFrame(oh_enc.transform(copy_df[['i_sex']]).todense(), 
                           columns=oh_enc.get_feature_names(),
                           index = copy_df.index)
    #getting necessary columns
    necessaryVars = copy_df[['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr',
       'i_hrm', 'i_hrf', "WOP", 'CPT']]
    #standardizing data
    scaler = preprocessing.StandardScaler().fit(necessaryVars)
    scaled_data = scaler.transform(necessaryVars)
    standardizedData = pd.DataFrame(scaled_data)
    standardizedData = standardizedData.join(dummies)
    standardizedData.columns = ['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr',
       'i_hrm', 'i_hrf', "WOP", 'CPT', 'x0_1', 'x0_2', 'x0_3']
    return standardizedData
# %%
clean_LDR_DATA = data_stand_ohenc_pipeline(LDR_DATA)
# %%
#clean_LDR_DATA = clean_LDR_DATA.sample(frac = 0.1)
# %%
clean_LDR_DATA.head()
# %%
clean_LDR_DATA.describe()
# %%
#WOP MODEL - creating training and testing sets
X_WOP = clean_LDR_DATA[['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr',
       'i_hrm', 'i_hrf', 'x0_1', 'x0_2', 'x0_3']]
Y_WOP = clean_LDR_DATA["WOP"]

X_WOP_train, X_WOP_test, Y_WOP_train, Y_WOP_test = train_test_split(X_WOP, Y_WOP, test_size=0.20, random_state=42)

# %%
X_WOP_train
# %%
Y_WOP_train
# %%
#WOP MODEL - training and testing model
ldr_wop_model = svm.LinearSVR(loss='squared_epsilon_insensitive', dual = False, max_iter=1200000)
ldr_wop_model.fit(X_WOP_train, Y_WOP_train)
print("Done fitting")
ldr_wop_model_score = ldr_wop_model.score(X_WOP_test, Y_WOP_test)
print("Score: ") 
print(ldr_wop_model_score)
#creating plot
Y_WOP_pred = ldr_wop_model.predict(X_WOP_test)
plt.xlabel("Actual WOP")
plt.ylabel("Predicted WOP")
plt.title("Actual vs Predicted WOP - LDR")
plt.scatter(Y_WOP_test, Y_WOP_pred, s=.1)
plt.plot(Y_WOP_test, Y_WOP_test, color = 'red', label="Actual = Predicted")
# %%
# %%
maxWOP = max(Y_WOP_test)
minWOP = min(Y_WOP_test)

clean_Y_WOP_pred = Y_WOP_pred.copy()
for i in range(len(clean_Y_WOP_pred)):
       if clean_Y_WOP_pred[i] > maxWOP:
              clean_Y_WOP_pred[i] = maxWOP
       elif clean_Y_WOP_pred[i] < minWOP:
              clean_Y_WOP_pred[i] = minWOP

clean_ldr_wop_r2score = r2_score(Y_WOP_test, clean_Y_WOP_pred)
print(clean_ldr_wop_r2score)

plt.xlabel("Actual WOP")
plt.ylabel("Clean Predicted WOP")
plt.title("Actual vs Clean Predicted WOP - LDR")
plt.scatter(Y_WOP_test, clean_Y_WOP_pred, s=.1)
plt.plot(Y_WOP_test, Y_WOP_test, color = 'red', label="Actual = Predicted")
# %%

# %%
#CPT MODEL - creating training and testing sets
X_CPT = clean_LDR_DATA[['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr',
       'i_hrm', 'i_hrf', 'x0_1', 'x0_2', 'x0_3']]
Y_CPT = clean_LDR_DATA["CPT"]

X_CPT_train, X_CPT_test, Y_CPT_train, Y_CPT_test = train_test_split(X_CPT, Y_CPT, test_size = 0.20, random_state = 42)
# %%
X_CPT_train
# %%
Y_CPT_train
# %%
#CPT MODEL - training and testing model
ldr_cpt_model = svm.LinearSVR(loss='squared_epsilon_insensitive', dual = False, max_iter=1200000)
ldr_cpt_model.fit(X_CPT_train, Y_CPT_train)
print("Done fitting")
ldr_cpt_model_score = ldr_cpt_model.score(X_CPT_test, Y_CPT_test)
print("Score: ")
print(ldr_cpt_model_score)

Y_CPT_pred = ldr_cpt_model.predict(X_CPT_test)
plt.xlabel("Actual CPT")
plt.ylabel("Predicted CPT")
plt.title("Actual vs Predicted CPT - LDR")
plt.scatter(Y_CPT_test, Y_CPT_pred, s=.1)
plt.plot(Y_CPT_test, Y_CPT_test, color = 'red', label="Actual = Predicted")
# %%
maxCPT = max(Y_CPT_test)
minCPT = min(Y_CPT_test)

clean_Y_CPT_pred = Y_CPT_pred.copy()
for i in range(len(clean_Y_CPT_pred)):
       if clean_Y_CPT_pred[i] > maxCPT:
              clean_Y_CPT_pred[i] = maxCPT
       elif clean_Y_CPT_pred[i] < minCPT:
              clean_Y_CPT_pred[i] = minCPT

clean_ldr_cpt_r2score = r2_score(Y_CPT_test, clean_Y_CPT_pred)
print(clean_ldr_cpt_r2score)

plt.xlabel("Actual CPT")
plt.ylabel("Clean Predicted CPT")
plt.title("Actual vs Clean Predicted CPT - LDR")
plt.scatter(Y_CPT_test, clean_Y_CPT_pred, s=.1)
plt.plot(Y_CPT_test, Y_CPT_test, color = 'red', label="Actual = Predicted")
# %%

# %%


###############################################################################
# Getting SDR dataset
###############################################################################
SDR_GDRIVE = 'dsdp_datasets/SDR'
# %%
###############################################################################
# Load dataset
###############################################################################
expPathSDR = path.join(BASE_PATH, SDR_GDRIVE, FILE_NAME)
SDR_DATA = pd.read_csv(expPathSDR)
print('* Dataset Path: {}'.format(expPathSDR))
print('* Dataset Dimensions: {}'.format(SDR_DATA.shape))
SDR_DATA.head()
# %%
clean_SDR_DATA = data_stand_ohenc_pipeline(SDR_DATA)
# %%
clean_SDR_DATA.head()
# %%
#WOP MODEL - creating training and testing sets
X_WOP_SDR = clean_SDR_DATA[['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr',
       'i_hrm', 'i_hrf', 'x0_1', 'x0_2', 'x0_3']]
Y_WOP_SDR = clean_SDR_DATA["WOP"]

X_WOP_SDR_train, X_WOP_SDR_test, Y_WOP_SDR_train, Y_WOP_SDR_test = train_test_split(X_WOP_SDR, Y_WOP_SDR, test_size=0.20, random_state=42)

# %%
X_WOP_SDR_train
# %%
Y_WOP_SDR_train
# %%
#WOP MODEL - training and testing model
sdr_wop_model = svm.LinearSVR(loss='squared_epsilon_insensitive', dual = False, max_iter=1200000)
sdr_wop_model.fit(X_WOP_SDR_train, Y_WOP_SDR_train)
print("Done fitting")
sdr_wop_model_score = sdr_wop_model.score(X_WOP_SDR_test, Y_WOP_SDR_test)
print("Score: ")
print(sdr_wop_model_score)

Y_WOP_SDR_pred = sdr_wop_model.predict(X_WOP_SDR_test)
plt.xlabel("Actual WOP")
plt.ylabel("Predicted WOP")
plt.title("Actual vs Predicted WOP - SDR")
plt.scatter(Y_WOP_SDR_test, Y_WOP_SDR_pred, s=.1)
plt.plot(Y_WOP_SDR_test, Y_WOP_SDR_test, color = 'red', label="Actual = Predicted")
# %%
maxWOP = max(Y_WOP_SDR_test)
minWOP = min(Y_WOP_SDR_test)

clean_Y_WOP_SDR_pred = Y_WOP_SDR_pred.copy()
for i in range(len(clean_Y_WOP_SDR_pred)):
       if clean_Y_WOP_SDR_pred[i] > maxWOP:
              clean_Y_WOP_SDR_pred[i] = maxWOP
       elif clean_Y_WOP_SDR_pred[i] < minWOP:
              clean_Y_WOP_SDR_pred[i] = minWOP

clean_sdr_wop_r2score = r2_score(Y_WOP_SDR_test, clean_Y_WOP_SDR_pred)
print(clean_sdr_wop_r2score)

plt.xlabel("Actual WOP")
plt.ylabel("Clean Predicted WOP")
plt.title("Actual vs Clean Predicted WOP - SDR")
plt.scatter(Y_WOP_SDR_test, clean_Y_WOP_SDR_pred, s=.1)
plt.plot(Y_WOP_SDR_test, Y_WOP_SDR_test, color = 'red', label="Actual = Predicted")
# %%
#CPT MODEL - creating training and testing sets
X_CPT_SDR = clean_SDR_DATA[['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr',
       'i_hrm', 'i_hrf', 'x0_1', 'x0_2', 'x0_3']]
Y_CPT_SDR = clean_SDR_DATA["CPT"]

X_CPT_SDR_train, X_CPT_SDR_test, Y_CPT_SDR_train, Y_CPT_SDR_test = train_test_split(X_CPT_SDR, Y_CPT_SDR, test_size = 0.20, random_state = 42)
# %%
X_CPT_SDR_train
# %%
Y_CPT_SDR_train
# %%
#CPT MODEL - training and testing model
sdr_cpt_model = svm.LinearSVR(loss='squared_epsilon_insensitive', dual = False, max_iter=1200000)
sdr_cpt_model.fit(X_CPT_SDR_train, Y_CPT_SDR_train)
print("Done fitting")
sdr_cpt_model_score = sdr_cpt_model.score(X_CPT_SDR_test, Y_CPT_SDR_test)
print("Score: ")
print(sdr_cpt_model_score)

Y_CPT_SDR_pred = sdr_cpt_model.predict(X_CPT_SDR_test)
plt.xlabel("Actual CPT")
plt.ylabel("Predicted CPT")
plt.title("Actual vs Predicted CPT - SDR")
plt.scatter(Y_CPT_SDR_test, Y_CPT_SDR_pred, s=.1)
plt.plot(Y_CPT_SDR_test, Y_CPT_SDR_test, color = 'red', label="Actual = Predicted")
# %%
# %%
maxCPT = max(Y_CPT_SDR_test)
minCPT = min(Y_CPT_SDR_test)

clean_Y_CPT_SDR_pred = Y_CPT_SDR_pred.copy()
for i in range(len(clean_Y_CPT_SDR_pred)):
       if clean_Y_CPT_SDR_pred[i] > maxCPT:
              clean_Y_CPT_SDR_pred[i] = maxCPT
       elif clean_Y_CPT_SDR_pred[i] < minCPT:
              clean_Y_CPT_SDR_pred[i] = minCPT

clean_sdr_cpt_r2score = r2_score(Y_CPT_SDR_test, clean_Y_CPT_SDR_pred)
print(clean_sdr_cpt_r2score)

plt.xlabel("Actual CPT")
plt.ylabel("Clean Predicted CPT")
plt.title("Actual vs Clean Predicted CPT - SDR")
plt.scatter(Y_CPT_SDR_test, clean_Y_CPT_SDR_pred, s=.1)
plt.plot(Y_CPT_SDR_test, Y_CPT_SDR_test, color = 'red', label="Actual = Predicted")


# %%
#getting adjusted R2 values for all models
# %%
def adjusted_r2_val(original_r2, predictions):
       n = len(predictions)
       p = 12
       adj_r2 = 1 - ((1 - original_r2)*(n - 1)/(n - p - 1))
       return adj_r2
# %%
adj_ldr_wop = adjusted_r2_val(ldr_wop_model_score, Y_WOP_pred)
print("Original R2: ", ldr_wop_model_score)
print("Adjusted R2: ", adj_ldr_wop)
# %%
adj_ldr_wop_clean = adjusted_r2_val(clean_ldr_wop_r2score, clean_Y_WOP_pred)
print("Original R2: ", clean_ldr_wop_r2score)
print("Adjusted R2: ", adj_ldr_wop_clean)
# %%
adj_ldr_cpt = adjusted_r2_val(ldr_cpt_model_score, Y_CPT_pred)
print("Original R2: ", ldr_cpt_model_score)
print("Adjusted R2: ", adj_ldr_cpt)
# %%
adj_ldr_cpt_clean = adjusted_r2_val(clean_ldr_cpt_r2score, clean_Y_CPT_pred)
print("Original R2: ", clean_ldr_cpt_r2score)
print("Adjusted R2: ", adj_ldr_cpt_clean)
# %%
adj_sdr_wop = adjusted_r2_val(sdr_wop_model_score, Y_WOP_SDR_pred)
print("Original R2: ", sdr_wop_model_score)
print("Adjusted R2: ", adj_sdr_wop)
# %%
adj_sdr_wop_clean = adjusted_r2_val(clean_sdr_wop_r2score, clean_Y_WOP_SDR_pred)
print("Original R2: ", clean_sdr_wop_r2score)
print("Adjusted R2: ", adj_sdr_wop_clean)
# %%
adj_sdr_cpt = adjusted_r2_val(sdr_cpt_model_score, Y_CPT_SDR_pred)
print("Original R2: ", sdr_cpt_model_score)
print("Adjusted R2: ", adj_sdr_cpt)
# %%
adj_sdr_cpt_clean = adjusted_r2_val(clean_sdr_cpt_r2score, clean_Y_CPT_SDR_pred)
print("Original R2: ", clean_sdr_cpt_r2score)
print("Adjusted R2: ", adj_sdr_cpt_clean)
# %%
#serializing models
# %%
import pickle
# %%
ldr_wop_f = 'ldr_wop_finalized_model.sav'
pickle.dump(ldr_wop_model, open(ldr_wop_f, 'wb'))
# %%
ldr_cpt_f = 'ldr_cpt_finalized_model.sav'
pickle.dump(ldr_cpt_model, open(ldr_cpt_f, 'wb'))
# %%
sdr_wop_f = 'sdr_wop_finalized_model.sav'
pickle.dump(sdr_wop_model, open(sdr_wop_f, 'wb'))
# %%
sdr_cpt_f = 'sdr_cpt_finalized_model.sav'
pickle.dump(sdr_cpt_model, open(sdr_cpt_f, 'wb'))
# %%
