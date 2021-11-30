# Discovery Program: Using Deep Learning to Predict Window of Protection and Cumulative Potential for Transmission for Mosquito Gene Drive

## Abstract

## Project Background

Across the last year, the UC Berkeley School of Public Health's Marshall Lab has generated datasets on simulations of the deployment of CRISPR/Cas9 genetically-modified mosquitos in the islands of São Tomé and Príncipe in equatorial Africa for research on mosquito borne illnesses.

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

We also want to split the data into training and testing sets and normalize the input values. Here, I use the scikit-learn MinMaxScaler() to normalize.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc=preprocessing.MinMaxScaler()
X_train = sc.fit_transform(X_train)

sc=preprocessing.MinMaxScaler()
X_test = sc.fit_transform(X_test)
```




Our group has been generating datasets on the expected results on deploying genetically-modified mosquitoes over the last year. One of the challenges, however, is to share our results with collaborators and stakeholders

These datasets were generated as part of a larger publication on the effects of simulating the releases of CRISPR/Cas9 linked and split drive genetic modifications in the islands of São Tomé and Príncipe in equatorial Africa. In these versions of the dataset, we consider the landscape as a fully-mixing panmictic population to focus on the inherent properties of the drives, without the spatial component of the migration of the mosquitos.

this is some text yeah

Some basic Git commands are:
```
git status
git add
git commit
```

This site was built using [GitHub Pages](https://pages.github.com/).

![This is an image](https://myoctocat.com/assets/images/base-octocat.svg)

- George Washington
- John Adams
- Thomas Jefferson
