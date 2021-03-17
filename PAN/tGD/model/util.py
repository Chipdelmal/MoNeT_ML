import sklearn.model_selection
import pandas as pd
import os

DATA_TYPES = {
    "i_hnf"     : 'int8'
    , "i_cac"   : 'int8'
    , "i_frc"   : 'int8'
    , "i_hrt"   : 'int8'
    , "i_ren"   : 'int8'
    , "i_res"   : 'int8'
    , "i_grp"   : 'int8'
    , "0.05"    : 'float64'
    , "0.1"     : 'float64'
    , "0.25"    : 'float64'
    , "0.5"     : 'float64'
    , "0.75"    : 'float64'
    , "0.9"     : 'float64'
    , "0.95"    : 'float64'
}

def get_data(key, path_to_data_directory):
    """
    Combines all CSV files with 'key' in their name 
    into a single DataFrame. Appends an extra column
    of integers, 'i_key', which denotes the file 
    where each data point came from. 

    Parameters:
    -----------

        key : string
            The substring to search for in each file name.

        path_to_data_directory : string
            A path to the directory of CSV files.

    Returns:
    --------
        A pandas dataframe of concatenated data.
    """
    data = []
    indx = 0
    for fname in os.listdir(path_to_data_directory):
        if fname.endswith('.csv') and key in fname:
            df = pd.read_csv(os.path.join(path_to_data_directory, fname), dtype=DATA_TYPES)
            df['i_key'] = indx
            data.append(df)
            indx += 1
    return pd.concat(data, ignore_index=True)

def stratified_split(X, y, *args, **kwargs):
    """
    Bins the target variable using 5 quantiles using
    a stratified shuffle split. Duplicate bins are 
    dropped.

    Parameters:
    -----------
        X : pandas.DataFrame
            The feature matrix.

        y : pandas.DataFrame
            The target vector.

        args : tuple
            Arguments for sklearn's StratifiedShuffleSplit function.

        kwargs : dict
            Keyword arguments for sklearn's StratifiedShuffleSplit function.

    Returns:
    --------
        Four items:
            1. X_train  - a pandas DataFrame of training data
            2. X_test   - a pandas DataFrame of testing data
            3. y_train  - a pandas categorical frame of training labels
            4. y_test   - a pandas categorical frame of testing labels
    """
    y = pd.qcut(y.values.reshape(-1), 5, duplicates='drop')
    sss = sklearn.model_selection.StratifiedShuffleSplit(*args, **kwargs)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test
