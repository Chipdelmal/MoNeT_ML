#%%
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import graphviz
# newly import
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from matplotlib.colors import ListedColormap, Normalize
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#%%
GDRIVE = 'LDR'
FILE_NAME = 'SCA_HLT_50Q_10T.csv'
BASE_PATH = '/Users/xinyu/Desktop/MoNeT_ML'
######################
###  Load dataset  ###
######################
expPath = path.join(BASE_PATH, GDRIVE, FILE_NAME)
DATA = pd.read_csv(expPath)
print('* Dataset Path: {}'.format(expPath))
print('* Dataset Dimensions: {}'.format(DATA.shape))
DATA.head()

#%%
##############################
###  Features we interest  ###
##############################
feature = DATA.loc[:, ['i_ren', 'i_res', 'i_rsg', 'i_gsv', 'i_fch', 'i_fcb', 'i_fcr', 'i_hrm', 'i_hrf','i_grp', 'i_mig']] 
CPT = DATA['CPT']
WOP = DATA['WOP']

check_y_data = DATA[['CPT','WOP']]
check_y_data

# %%
#######################
###  Split dataset  ###
#######################
WOP_x_train, WOP_x_test, WOP_train, WOP_test = train_test_split(feature, WOP, test_size=0.5, random_state=42)
CPT_x_train, CPT_x_test, CPT_train, CPT_test = train_test_split(feature, CPT, test_size=0.5, random_state=42)
# %%
######  Decision Tree Model  ######
decision_tree_model = DecisionTreeRegressor(max_depth=13,random_state=42,splitter="random")
# decision_tree_model.fit(x_train, CPT_train)
decision_tree_model.fit(WOP_x_train, WOP_train)
wop_hat_dec = decision_tree_model.predict(WOP_x_test) # numpy array 
wop_hat_dec

#%%
plt.figure(figsize=(10,8))
plt.xlabel("Actual WOP",fontsize=12)
plt.ylabel("Predicted WOP",fontsize=12)
plt.title("LDR: (Decision Tree) Actual vs Predicted WOP",fontsize=18)
plt.scatter(WOP_test, wop_hat_dec, s=20, edgecolor="gray", c="yellowgreen")
plt.plot(WOP_test, WOP_test, color = "red", label="Actual = Predicted")
plt.legend()
plt.savefig("LDR_Decision_Tree_WOP.png")
#%%
dt_training_accuracy = decision_tree_model.score(WOP_x_train, WOP_train)
dt_test_accuracy = decision_tree_model.score(WOP_x_test, WOP_test)
dt_training_accuracy, dt_test_accuracy
#%%
######  Random Forest Model  ###### 
random_forest_model = RandomForestRegressor(n_estimators=65, min_samples_split=10,max_depth=None)
random_forest_model.fit(WOP_x_train, CPT_train)
cpt_hat_ran = random_forest_model.predict(WOP_x_test)

#%%
plt.figure(figsize=(10,8))
plt.xlabel("Actual CPT",fontsize=14)
plt.ylabel("Predicted CPT",fontsize=14)
plt.title("LDR: (Random Forest) Actual vs Predicted CPT",fontsize=18)
plt.scatter(CPT_test, cpt_hat_ran, s=20, edgecolor="black", c="lightblue")
plt.plot(CPT_test, CPT_test, color = "red", label="Actual = Predicted")
plt.legend()
plt.savefig("LDR_Random_Forest_CPT.png")

#%%
rf_training_accuracy = random_forest_model.score(CPT_x_train, CPT_train)
rf_test_accuracy = random_forest_model.score(CPT_x_test, CPT_test)
rf_training_accuracy, rf_test_accuracy

# %%
feature_namees = feature.columns
feature_namees
# %%
feature_importance_dc = pd.DataFrame(decision_tree_model.feature_importances_, index = feature_namees).sort_values(0, ascending=False)
feature_importance_dc
# %%
feature_importance_rf = pd.DataFrame(random_forest_model.feature_importances_, index = feature_namees).sort_values(0, ascending=False)
feature_importance_rf
# %%
pd.concat([feature_importance_dc, feature_importance_rf], axis=1, ignore_index=True)
# %%
feature_importance_dc.plot(kind='bar')
plt.title("LDR: (Decision Tree) Feature Importances ",fontsize=16)
# %%
import pickle
with open('DecisionTreeRegressor_LDR.sav', 'wb') as f:
    pickle.dump(decision_tree_model,f)
# %%
