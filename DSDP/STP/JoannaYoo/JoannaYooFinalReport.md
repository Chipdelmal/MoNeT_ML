# DSDP: Using Support Vector Machines (SVM) to Predict Window of Protection and Cumulative Potential for Transmission for Mosquito Gene Drive

## By: Joanna Yoo

## Objective:
The main objective was to train a machine learning model to predict the effectiveness of genetially modified mosquitoes in decreasing the frequency of mosquito-transmitted diseases. Datasets were provided by UC Berkeley Marshall Lab. I chose to try out an SVM model (specifically the LinearSVR implementation from scikit-learn) to train 2 separate models for predicting the Window of Protection (WOP) and the Cumulative Potential for Transmission (CPT) using the LDR and SDR datasets provided by the lab.

## Data Cleaning:
Before training my model, we first needed to clean the data. More specifically, we needed to one-hot-encode the 'i_sex' column and make sure that our data was either standardizes or normalized. For the SVM model, I decided it would be better to standardize the data. In order to be able to reuse this cleaning process for the SDR dataset, I decided to group all of the cleaning steps into the function below:

```
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
```
## Building the Model
For my model, I ended up using all of the columns to train my model. I first started out using Scikit-Learn's svm SVR (Support Vector Regression) implementatin and setting the kernel to linear. However, I realized that the dataset was too big for the SVR implementation, so I switched to the LinearSVR implementation, which only takes into account a linear kernel and is much faster. For the different parameters, I set the loss parameter to squared epsilon-insensitive loss, which is equivalent to the L2 loss. I also set the dual parameter to False since this dataset had more samples than features

## Final Results
After using my models to visualize the predicted data points against the actual data points in the test set, I realized that there were these tails in the visualizations that corresponded to predicted values that fell out of the bounds of the original values. Therefore, in order to see how much the score would improve if the tails were fixed, I got the R2 score for both the original predictions and then the clean predictions where the values were adjusted so that they were in the original range.

LDR - WOP Model
![image](/FinalFigures/LDR_WOP/original.png)  
Original R2:  0.7469958852839678  
Adjusted R2:  0.7469806697370158  
<br />
LDR - WOP Model without Tails
![image](/FinalFigures/LDR_WOP/clean.png)  
Original R2:  0.7811917499441575  
Adjusted R2:  0.7811785909202187  
<br />
LDR - CPT Model
![image](/FinalFigures/LDR_CPT/original.png)  
Original R2:  0.7423267574819259  
Adjusted R2:  0.7423112611358519  
<br />
LDR - CPT Model without Tails
![image](/FinalFigures/LDR_CPT/clean.png)  
Original R2:  0.7821797082998418  
Adjusted R2:  0.7821666086912479  
<br />
SDR - WOP Model
![image](/FinalFigures/SDR_WOP/original.png)  
Original R2:  0.7821797082998418  
Adjusted R2:  0.7821666086912479  
<br />
SDR - WOP Model without Tails
![image](/FinalFigures/SDR_WOP/clean.png)  
Original R2:  0.6786951157701291  
Adjusted R2:  0.6786406457749217  
<br />
SDR - CPT Model
![image](/FinalFigures/SDR_CPT/original.png)  
Original R2:  0.6075402540119055  
Adjusted R2:  0.6074737213149801  
<br />
SDR - CPT Model without Tails
![image](/FinalFigures/SDR_CPT/clean.png)  
Original R2:  0.6594140205513355  
Adjusted R2:  0.6593562818813717  

