"""
Name: PYF_Preprocess
Description: A script that cleans and transforms the PYF HLT data to train the LightWeight POE and WOP ML Model.
"""
import pandas as pd
import os
import os.path as path
import sys
################################################
# Setup constants and paths
################################################
DATA_DIR_PATH = 'drive/Shareddrives/AI Mosquitos/Data'
CLN_DATA_DIR_PATH = 'drive/Shareddrives/AI Mosquitos/Data/Cln'
FILE_NAME_FMT = 'HLT_{0}_{1}_qnt.csv'
DATA_PERCENTILS = ['50', '75', '90']
TARGET_VARS = {
    'POE': ['POE'],
    'WOP': ['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95']
}
################################################
# Load dataset
################################################
USR = sys.argv[1]
LND = sys.argv[2]
BASE_DIR = LND
DATA_DIR_PATH = path.join(BASE_DIR, 'Data')
CLN_DATA_DIR_PATH = path.join(BASE_DIR, 'Clean')

metric = sys.argv[3]
quantile = sys.argv[4]
csv_path = path.join(DATA_DIR_PATH, FILE_NAME_FMT.format(metric, quantile))
print('Loading ' + csv_path)
df = pd.read_csv(csv_path)
# ################################################
# # Transform
# ################################################
def dropFeatures(df, extra=[]):
  columns = ['i_grp'] + extra
  return df.drop(columns=columns)
def normalize(df):
  cpy = df.copy()
  cpy.loc[:, 'i_res'] = cpy['i_res'] / 100.0
  cpy.loc[:, 'i_mad'] = cpy['i_mad']  / 100.0
  cpy.loc[:, 'i_mat'] = cpy['i_mat'] / 100.0
  return cpy
def extend(df):
  cpy = df.copy()
  cpy["i_nmosquitos"] = cpy["i_res"] * cpy["i_ren"] 
  return cpy

# Drop also POF in POE dataset
if metric == 'POE':
    extra = ['POF']
else:
    extra = []

# Apply
df = extend(normalize(dropFeatures(df, extra)))
# ################################################
# # Save dataset
# ################################################
if (not path.isdir(CLN_DATA_DIR_PATH)):
    os.mkdir(CLN_DATA_DIR_PATH)

df.to_csv(os.path.join(CLN_DATA_DIR_PATH, FILE_NAME_FMT.format(metric, quantile)), index=False)