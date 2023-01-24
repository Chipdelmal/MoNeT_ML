#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
from keras.models import load_model
import compress_pickle as pkl
import constants as cst

###############################################################################
# Load Model
###############################################################################
def loadModel(
        AOI, THS, MTR, MDL, 
        QNT=None, PATH_MDL=cst.PATH_MDL
    ):
    # Check if model is QNT or regular (reps) ---------------------------------
    if QNT:
        mdlName = f'{AOI}_{QNT}Q_{float(THS)*100:.0f}T_{MTR}-{MDL}-MLR'
    else:
        mdlName = f'{AOI}_{float(THS)*100:.0f}T_{MTR}-{MDL}-MLR'
    # Check if model is keras or scikit-learn ---------------------------------
    if MDL=='krs':
        mdlPath = path.join(PATH_MDL, mdlName)
        rf = load_model(mdlPath)
    else:
        mdlPath = path.join(PATH_MDL, mdlName+'.pkl')
        rf = pkl.load(mdlPath)
    return rf