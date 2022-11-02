#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from dash import Dash
from flask import Flask
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

import numpy as np
from treeinterpreter import treeinterpreter as ti
import auxiliary as aux


debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
###############################################################################
# Setup Dash App
###############################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = 'pgSIT Cost Effectiveness'

###############################################################################
# Run Model
###############################################################################
rf = aux.loadModel('HLT', '0.1', 'WOP')
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
print(prediction)

###############################################################################
# Run Dash App
###############################################################################
app.layout = dbc.Container([ 
    dbc.Row(
        dbc.Col(
            html.H2(str(prediction[0][0])), 
            width={'size': 12, 'offset': 0, 'order': 0}
        ), style = {'textAlign': 'center', 'paddingBottom': '1%'}
    )
])

# @app.callback()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=debug)
