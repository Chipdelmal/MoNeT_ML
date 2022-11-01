#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from os import path
import numpy as np
import pandas as pd
import plotly.express as px
import compress_pickle as pkl
from treeinterpreter import treeinterpreter as ti

(AOI, THS, MTR) = ('HLT', '0.1', 'WOP')
PATH_BASE = '/Users/sanchez.hmsc/odrive/Mega/WorkSync/MLModels/PAN/PGS/'
###############################################################################
# Load Model
###############################################################################
mdlName = f'{AOI}_{float(THS)*100:.0f}T_{MTR}-MLR.pkl'
mdlPath = path.join(PATH_BASE, mdlName)
rf = pkl.load(mdlPath)
###############################################################################
# Evaluate Model
###############################################################################
probeX = (
    ('ren', 30),
    ('rer', 30),
    ('rei', 7),
    ('pct', 1),
    ('pmd', 1),
    ('mfr', 0),
    ('mtf', 1),
    ('fvb', 0),
)
FEATS = [i[0] for i in probeX]
# Evaluate models at probe point ----------------------------------------------
vct = np.array([[i[1] for i in probeX]])
(prediction, bias, contributions) = ti.predict(rf, vct)
prediction

fig = px.bar(contributions[0])
fig.write_html(path.join(PATH_BASE, 'bars.html'))