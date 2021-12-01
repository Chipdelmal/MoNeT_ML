# Discovery Program: Using Deep Learning to Predict Window of Protection and Cumulative Potential for Transmission for Mosquito Gene Drive

## By: Ayden Salazar

## Abstract

Across the last year, the UC Berkeley School of Public Health's Marshall Lab has generated datasets on simulations of the deployment of CRISPR/Cas9 genetically-modified mosquitos in the islands of São Tomé and Príncipe in equatorial Africa for research on mosquito-borne illnesses.

## Objectives

The overall goal is to produce machine learning models that are trained on the summary statistics from the simulation data. These models should describe the behavior of the system. In the case of my role, the task is to train a deep learning model (I chose a neural network) that predicts the Window of Protection (WOP) and Cumulative Potential for Transmission (CPT) variables in the simulation data. There are two major datasets: LDR and SDR. Since there are two variable from each we want to predict, I will be training 4 models in total.


# Data Cleaning and Exploratory Data Analysis

First and foremost, since the sex column in the datasets, which designates the sex of the released transgenic mosquitos, is categorical, we must convert this to a numerical value via one-hot encoding.

```
# Convert sex column into one-hot encoded columns
for i, df in enumerate(df_list):
    if 'i_sex' in df.columns:

        # assign one-hots
        oh = pd.get_dummies(df['i_sex'], prefix='i_sex')
        new_df = pd.concat([df, oh], axis=1).drop(['i_sex'], axis=1)

        df_list[i] = new_df
```

We also want to split the data into training and testing sets and normalize the input values. Here, I use the scikit-learn function MinMaxScaler() to normalize.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc=preprocessing.MinMaxScaler()
X_train = sc.fit_transform(X_train)

sc=preprocessing.MinMaxScaler()
X_test = sc.fit_transform(X_test)
```

# Construction of Neural Network

To construct the neural network, I used the Tensorflow/Keras Python deep learning library.


For predicting the CPT, I created a hidden layer of size 30 with an ReLu activation function, one other hidden layer of size 5 (also with ReLu), and an output layer with a Sigmoid activation function. This setup brought the best results:
```
num_dim = X_train.shape[1]
print("Num dim:", num_dim)
model = Sequential()
model.add(Dense(30, input_dim=num_dim, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

Next, I set the optimizer for the neural network to be stochastic gradient descent with a learning rate of .01. I then fit the model to the training data with 100 epochs and a batch size of 500 for the hyperparameters.

```
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) # optimizer adam works

# fit model
print("LEN of X_TRAIN:", X_test.shape, "len of y_train:", y_test.shape)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=500) # works well with 500 batch size
```

# Results/Data Visualizations

Finally, I plotted the results. The results for the actual y values vs predicted y value for both CPT and WOP are below, as well as the code:

```
y_pred = model.predict(X_test)
mean_squared_error(y_pred, y_test)  
# %%
plt.xlabel("Actual CPT")
plt.ylabel("Predicted CPT")
plt.title("Actual vs Predicted CPT using Batch-Trained Neural Network on All Features")
plt.scatter(y_test, y_pred, s=.1)
plt.plot(y_test, y_test, color = 'red', label="Actual = Predicted")
plt.legend()
```

#### Actual vs Predicted CPT using Batch-Trained Neural Network on All SDR Features
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/AydenSalazar/DataVisualizationsAydenSalazar/SDR_CPT.jpg)

#### Actual vs Predicted WOP using Batch-Trained Neural Network on All SDR Features
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/AydenSalazar/DataVisualizationsAydenSalazar/SDR_WOP.jpg)

#### Actual vs Predicted CPT using Batch-Trained Neural Network on All LDR Features
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/AydenSalazar/DataVisualizationsAydenSalazar/LDR_CPT.jpg)

#### Actual vs Predicted WOP using Batch-Trained Neural Network on All LDR Features
![This is an image](https://github.com/Chipdelmal/MoNeT_ML/blob/main/DSDP/STP/AydenSalazar/DataVisualizationsAydenSalazar/LDR_WOP.jpg)
