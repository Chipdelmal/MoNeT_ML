# Discovery Program: Using decision tree and random forest regression to predict Window of Protection and Cumulative Potential for Transmission for Genetically Modified Mosquitos

## By: Xingli Yu

## Abstract & Objectives

Genome engineering-based mosquito control strategies including gene drives are promising approaches to mitigating the transmission risk of mosquito-borne diseases such as dengue, malaria, and Zika. The UC Berkeley School of Public Health's Marshall Lab has been generating datasets on the expected results of deploying genetically-modified mosquitoes on the islands of São Tomé and Príncipe in equatorial Africa over the last year. However, the gene drive datasets remain challenging to share with collaborators and stakeholders. In this project, I explore the use of a decision tree and random forest model in the mosquito gene drive datasets (both LDR and SDR) to predict the effectiveness, particularly in regards to the Window of Protection (WOP) and the Cumulative Potential for Transmission (CPT) variables of genetically modified mosquitoes in the control of mosquito-borne diseases.


# Feature Scaling and Training Model 

First of all, I split the data into training and testing sets using a 50:50 ratio from sci-kit learn's “train_test_split” function. I use Scikit-Learn's StandardScaler class to scale our data. However, I did not normalize them since the accuracy for both models did not change significantly from my preliminary attempts.

```
# fit model
WOP_x_train, WOP_x_test, WOP_train, WOP_test = train_test_split(feature, WOP, test_size=0.5, random_state=42)
CPT_x_train, CPT_x_test, CPT_train, CPT_test = train_test_split(feature, CPT, test_size=0.5, random_state=42)
 
sc = StandardScaler()
WOP_x_train = sc.fit_transform(WOP_x_train)
WOP_x_test = sc.transform(WOP_x_test)
```

I fit the data using sci-kit learn’s decision tree and random forest regression model, adjusting different parameters (max_depth, n_estimators, etc.) to optimize the accuracy. 

```
decision_tree_model = DecisionTreeRegressor(max_depth=13,random_state=42,splitter="random")
random_forest_model = RandomForestRegressor(n_estimators=65, min_samples_split=10,max_depth=None)
```

# Model Evaluation

I use Scikit-Learn’s K-fold cross-validation feature to evaluate the performance of the models. I chose 5 folds due to efficiency.
For LDR datasets:
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/XingliYu/Figures/LDR_Evaluation.jpg)

For SDR datasets:
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/XingliYu/Figures/SDR_Evaluation.jpg)

# Final Models (Actual vs Predicted)

The results for the actual values vs predicted values for “CPT” and “WOP” are shown below:
#### For LDR datasets:
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/XingliYu/Figures/LDR_Models.jpg)

#### For SDR datasets:
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/XingliYu/Figures/SDR_Models.jpg)

# Data Visualizations

Utilizing the benefits of decision trees and random forest models, I also explored the “.feature_importances_” function. The same features (“i_fch” and “i_fcb”) are detected as the most important features using both methods. 

#### For LDR datasets:
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/XingliYu/Figures/LDR_Feature_Importances.jpg)

#### For SDR datasets:
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/XingliYu/Figures/SDR_Feature_Importances.jpg)
