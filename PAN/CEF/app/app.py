#!/usr/bin/python
# -*- coding: utf-8 -*-

from dash import Dash
from flask import Flask
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

import numpy as np
from treeinterpreter import treeinterpreter as ti
import auxiliary as aux


###############################################################################
# Setup Dash App
###############################################################################
server = Flask(__name__)
app = Dash(server=server, external_stylesheets=[dbc.themes.FLATLY])
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

if __name__=='__main__':
    app.run_server()
    
#     fig = px.bar(contributions[0])
#     fig.write_html(path.join(PATH_BASE, 'bars.html'))