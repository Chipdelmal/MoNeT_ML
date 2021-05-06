import sys
import os.path as path
import numpy as np
import joblib

from PYF_Model import LWModel

QUANTILES = ['50', '75', '90']
TARGET_VARS = {
    'POE': ['POE'],
    'WOP': ['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95']
}

LABELS = ['LOW', 'MID', 'HIGH']

def inputOrDefault(msg, default):
    inp_str = input(msg)
    if (len(inp_str) == 0):
        return default
    else:
        return inp_str

if __name__ == "__main__":
    print("Welcome to the LW-PYF-ML-Shell!")
    print("A simple CLI tool to make predictions\n\n")

    # Load Model
    BASE_DIR_PATH = sys.argv[1]
    MODELS_DIR_PATH = path.join(BASE_DIR_PATH, 'Model')
    pyf_model = joblib.load(path.join(MODELS_DIR_PATH, 'lw_pyf_model.model'))

    
    while True:
        opt = str(input("Do you want to make a prediction (Y/n)? "))
        if opt == 'n' or opt == 'N':
            print("Bye bye!")
            break

        # Get Envirionment Input Option
        print("You'll shortly enter (pop, ren, res, mad, mat)")
        
        # Get Hyper Parameters
        i_pop = float(inputOrDefault("pop (32): ", 32))
        i_ren = float(inputOrDefault("ren (18): ", 18))
        i_res = float(inputOrDefault("res (100): ", 100))
        i_mad = float(inputOrDefault("mad (25): ", 25))
        i_mat = float(inputOrDefault("mat (30): ", 30))

        x = [i_pop, i_ren, i_res, i_mad, i_mat]

        res = pyf_model.predict(x)

        print('---------------------------------------')
        print('Results:')
        print('---------------------------------------')
        print('POE: {} with {}%% confidence'.format(res['POE'][0], res['POE'][1]))
        print('\n')

        for thresh in TARGET_VARS['WOP']:
            for qnt in QUANTILES:
                print('Q{} - WOP[{}]: {}'.format(qnt, thresh, res['WOP'][thresh][qnt][0]))
            print('')
        print('---------------------------------------')