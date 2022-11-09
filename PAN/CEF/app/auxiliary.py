#!/usr/bin/python
# -*- coding: utf-8 -*-

from os import path
import compress_pickle as pkl
import constants as cst

###############################################################################
# Load Model
###############################################################################
def loadModel(AOI, THS, MTR, PATH_MDL=cst.PATH_MDL):
    mdlName = f'{AOI}_{float(THS)*100:.0f}T_{MTR}-MLR.pkl'
    mdlPath = path.join(PATH_MDL, mdlName)
    rf = pkl.load(mdlPath)
    return rf

def loadModelNew(AOI, THS, MTR, MDL, PATH_MDL=cst.PATH_MDL):
    mdlName = f'{AOI}_{float(THS)*100:.0f}T_{MTR}-{MDL}-MLR.pkl'
    mdlPath = path.join(PATH_MDL, mdlName)
    rf = pkl.load(mdlPath)
    return rf