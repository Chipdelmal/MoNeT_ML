#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
import compress_pickle as pkl
import constants as cst

###############################################################################
# Load Model
###############################################################################
def loadModel(AOI, THS, MTR, MDL, QNT=None, PATH_MDL=cst.PATH_MDL):
    if QNT:
        mdlName = f'{AOI}_{QNT}Q_{float(THS)*100:.0f}T_{MTR}-{MDL}-MLR.pkl'
    else:
        mdlName = f'{AOI}_{float(THS)*100:.0f}T_{MTR}-{MDL}-MLR.pkl'
    mdlPath = path.join(PATH_MDL, mdlName)
    rf = pkl.load(mdlPath)
    return rf