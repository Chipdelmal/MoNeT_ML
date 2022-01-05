#%%
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi
import graphviz
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

#%%
GDRIVE = 'SDR'
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

#########################
###  Feature Scaling  ###
#########################
sc = StandardScaler() 
WOP_x_train = sc.fit_transform(WOP_x_train) 
WOP_x_test = sc.transform(WOP_x_test)
CPT_x_train = sc.fit_transform(CPT_x_train) 
CPT_x_test = sc.transform(CPT_x_test)
# %%
######  Decision Tree Model: WOP  ######
SDR_decision_tree_model_wop = DecisionTreeRegressor(max_depth=13,random_state=42,splitter="random")
SDR_decision_tree_model_wop.fit(WOP_x_train, WOP_train)
wop_hat_dec = SDR_decision_tree_model_wop.predict(WOP_x_test)  

#%%
plt.figure(figsize=(10,8))
plt.xlabel("Actual WOP",fontsize=12)
plt.ylabel("Predicted WOP",fontsize=12)
plt.title("SDR: (Decision Tree) Actual vs Predicted WOP",fontsize=18)
plt.scatter(WOP_test, wop_hat_dec, s=20, edgecolor="gray", c="yellowgreen")
plt.plot(WOP_test, WOP_test, color = "red", label="Actual = Predicted")
plt.legend()
plt.savefig("SDR_Decision_Tree_WOP.jpg")

######  Evaluation of Decision Tree Model for WOP  ######
#%%
WOP_dt_scores_mse = cross_val_score(SDR_decision_tree_model_wop, WOP_x_train, WOP_train, scoring="neg_mean_squared_error", cv=5)
wop_dt_rmse_scores = pd.Series(np.sqrt(-WOP_dt_scores_mse))
WOP_dt_scores_r2 = pd.Series(cross_val_score(SDR_decision_tree_model_wop, WOP_x_train, WOP_train, scoring="r2", cv=5))

#%%
######  Save Decision Tree Model  ######
with open('SDR_DecisionTreeRegressor_WOP.sav', 'wb') as f1:
    pickle.dump(SDR_decision_tree_model_wop,f1)
#%%

######  Random Forest Model: WOP  ###### 
SDR_random_forest_model_wop = RandomForestRegressor(n_estimators=65, min_samples_split=10,max_depth=None)
SDR_random_forest_model_wop.fit(WOP_x_train, WOP_train)
wop_hat_ran = SDR_random_forest_model_wop.predict(WOP_x_test)

#%%
plt.figure(figsize=(10,8))
plt.xlabel("Actual WOP",fontsize=14)
plt.ylabel("Predicted WOP",fontsize=14)
plt.title("SDR: (Random Forest) Actual vs Predicted WOP",fontsize=18)
plt.scatter(WOP_test, wop_hat_ran, s=20, edgecolor="black", c="lightblue")
plt.plot(WOP_test, WOP_test, color = "red", label="Actual = Predicted")
plt.legend()
plt.savefig("SDR_Random_Forest_WOP.jpg")

#%%
WOP_rf_scores_mse = cross_val_score(SDR_random_forest_model_wop, WOP_x_train, WOP_train, scoring="neg_mean_squared_error", cv=5)
wop_rf_rmse_scores = pd.Series(np.sqrt(-WOP_rf_scores_mse))
WOP_rf_scores_r2 = pd.Series(cross_val_score(SDR_random_forest_model_wop, WOP_x_train, WOP_train, scoring="r2", cv=5))
#%%
wop_train_accuracy_df = pd.DataFrame(columns=['DecisionTree_RMSE', 'DecisionTree_R2', 'RandomForest_RMSE', 'RandomForest_R2'])
wop_train_accuracy_df = pd.concat([wop_dt_rmse_scores, WOP_dt_scores_r2, wop_rf_rmse_scores, WOP_rf_scores_r2], axis=1, ignore_index=True).rename(columns={0: 'DecisionTree_RMSE', 1: 'DecisionTree_R2', 2: 'RandomForest_RMSE', 3:'RandomForest_R2'})
# wop_train_accuracy_df = pd.concat([wop_dt_rmse_scores, WOP_dt_scores_r2, wop_rf_rmse_scores, WOP_rf_scores_r2], axis=1, ignore_index=True).rename(columns={0: 'DecisionTree_RMSE', 1: 'DecisionTree_R2', 2: 'RandomForest_RMSE', 3:'RandomForest_R2'})
wop_train_accuracy_df.style.set_caption("WOP: Evaluation Models Performance")
wop_train_accuracy_df

#%%
######  Save Random Forest Model  ######
with open('SDR_RandomForestRegressor_WOP.sav', 'wb') as f2:
    pickle.dump(SDR_random_forest_model_wop,f2)

dfi.export(wop_train_accuracy_df,"SDR_wop_evaluation_df.jpg")

# %%
feature_namees = feature.columns
# %%
feature_importance_dc_wop = pd.DataFrame(SDR_decision_tree_model_wop.feature_importances_, index = feature_namees).sort_values(0, ascending=False)
feature_importance_rf_wop = pd.DataFrame(SDR_random_forest_model_wop.feature_importances_, index = feature_namees).sort_values(0, ascending=False)
# %%
DDF = pd.concat([feature_importance_dc_wop, feature_importance_rf_wop], axis=1, ignore_index=True).rename(columns={0: 'Decision Tree', 1: 'Random Forest'})
DDF
# %%
plt.figure(figsize=(10,8))
DDF.plot(kind='bar',fontsize=10)
plt.xticks(rotation=45)
plt.title("(SDR)Feature Importances WOP: Decision Tree vs. Random Forest",fontsize=12)
plt.xlabel("Features")
plt.ylabel("Feature Importance %")
plt.savefig("SDR_Feature_Importances_WOP.jpg")
# %%
######  Decision Tree Model: CPT  ######
SDR_decision_tree_model_cpt = DecisionTreeRegressor(max_depth=13,random_state=42,splitter="random")
SDR_decision_tree_model_cpt.fit(CPT_x_train, CPT_train)
cpt_hat_dec = SDR_decision_tree_model_cpt.predict(CPT_x_test)  

#%%
plt.figure(figsize=(10,8))
plt.xlabel("Actual CPT",fontsize=12)
plt.ylabel("Predicted CPT",fontsize=12)
plt.title("SDR: (Decision Tree) Actual vs Predicted CPT",fontsize=18)
plt.scatter(CPT_test, cpt_hat_dec, s=20, edgecolor="gray", c="yellowgreen")
plt.plot(CPT_test, CPT_test, color = "red", label="Actual = Predicted")
plt.legend()
plt.savefig("SDR_Decision_Tree_CPT.jpg")
#%%
######  Evaluation of Decision Tree Model for CPT  ######
#%%
CPT_dt_scores_mse = cross_val_score(SDR_decision_tree_model_cpt, CPT_x_train, CPT_train, scoring="neg_mean_squared_error", cv=5)
cpt_dt_rmse_scores = pd.Series(np.sqrt(-CPT_dt_scores_mse))
CPT_dt_scores_r2 = pd.Series(cross_val_score(SDR_decision_tree_model_cpt, CPT_x_train, CPT_train, scoring="r2", cv=5))
cpt_train_accuracy_df = pd.concat([cpt_dt_rmse_scores, CPT_dt_scores_r2], axis=1, ignore_index=True).rename(columns={0: 'RMSE', 1: 'R2'})
cpt_train_accuracy_df

######  Save Decision Tree Model  ######
with open('SDR_DecisionTreeRegressor_CPT.sav', 'wb') as f3:
    pickle.dump(SDR_decision_tree_model_cpt,f3)
#%%

######  Random Forest Model: CPT  ###### 
SDR_random_forest_model_cpt = RandomForestRegressor(n_estimators=65, min_samples_split=10,max_depth=None)
SDR_random_forest_model_cpt.fit(CPT_x_train, CPT_train)
cpt_hat_ran = SDR_random_forest_model_cpt.predict(CPT_x_test)

#%%
plt.figure(figsize=(10,8))
plt.xlabel("Actual CPT",fontsize=14)
plt.ylabel("Predicted CPT",fontsize=14)
plt.title("SDR: (Random Forest) Actual vs Predicted CPT",fontsize=18)
plt.scatter(CPT_test, cpt_hat_ran, s=20, edgecolor="black", c="lightblue")
plt.plot(CPT_test, CPT_test, color = "red", label="Actual = Predicted")
plt.legend()
plt.savefig("SDR_Random_Forest_CPT.jpg")
#%%
CPT_rf_scores_mse = cross_val_score(SDR_random_forest_model_cpt, CPT_x_train, CPT_train, scoring="neg_mean_squared_error", cv=5)
cpt_rf_rmse_scores = pd.Series(np.sqrt(-CPT_rf_scores_mse))
CPT_rf_scores_r2 = pd.Series(cross_val_score(SDR_random_forest_model_cpt, CPT_x_train, CPT_train, scoring="r2", cv=5))
#%%
#%%
cpt_train_accuracy_df = pd.concat([cpt_dt_rmse_scores, CPT_dt_scores_r2, cpt_rf_rmse_scores, CPT_rf_scores_r2], axis=1, ignore_index=True).rename(columns={0: 'DecisionTree_RMSE', 1: 'DecisionTree_R2', 2: 'RandomForest_RMSE', 3:'RandomForest_R2'})
cpt_train_accuracy_df.style.set_caption("CPT: Evaluation Models' Performance")
cpt_train_accuracy_df

#%%
######  Save Random Forest Model  ######
with open('SDR_RandomForestRegressor_CPT.sav', 'wb') as f4:
    pickle.dump(SDR_random_forest_model_cpt,f4)

dfi.export(cpt_train_accuracy_df,"SDR_cpt_evaluation_df.jpg")
# %%
feature_importance_dc_cpt = pd.DataFrame(SDR_decision_tree_model_cpt.feature_importances_, index = feature_namees).sort_values(0, ascending=False)
feature_importance_rf_cpt = pd.DataFrame(SDR_random_forest_model_cpt.feature_importances_, index = feature_namees).sort_values(0, ascending=False)
# %%
DDFF = pd.concat([feature_importance_dc_cpt, feature_importance_rf_cpt], axis=1, ignore_index=True).rename(columns={0: 'Decision Tree', 1: 'Random Forest'})
DDFF
# %%
plt.figure(figsize=(10,8))
DDFF.plot(kind='bar',fontsize=10)
plt.xticks(rotation=45)
plt.title("(SDR)Feature Importances CPT: Decision Tree vs. Random Forest",fontsize=12)
plt.xlabel("Features")
plt.ylabel("Feature Importance %")
plt.savefig("SDR_Feature_Importances_CPT.jpg")
# %%
