#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from dash import dcc
from dash import html
from dash import Dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
from treeinterpreter import treeinterpreter as ti

import layouts as lay
import auxiliary as aux


# debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
###############################################################################
# Setup Dash App
###############################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = 'pgSIT Cost Effectiveness'

###############################################################################
# Run Model
###############################################################################
RF = aux.loadModel('HLT', '0.1', 'WOP')
probe = (
    ('ren', 30),
    ('rer', 30),
    ('rei', 7),
    ('pct', 1),
    ('pmd', 1),
    ('mfr', 0),
    ('mtf', 1),
    ('fvb', 0),
)
FEATS = [i[0] for i in probe]
# Evaluate models at probe point ----------------------------------------------
vct = np.array([[i[1] for i in probe]])
# (prediction, bias, contributions) = ti.predict(rf, vct)
# pred = int(prediction[0][0])
(prediction, bias, contributions) = (RF.predict(vct), None, None)
pred = prediction[0]

###############################################################################
# Generate Layout
###############################################################################
app.layout = dbc.Container([ 
    dbc.Row(
        dbc.Col(
            dbc.Container([ 
                html.H2("psSIT Cost Effectiveness"),
                dbc.Col(html.Hr())
            ]),
            width={'size': 12, 'offset': 0, 'order': 0}
        ), style = {'textAlign': 'center', 'paddingBottom': '1%'}
    ),
    dbc.Row(        
        dbc.Col(
            dbc.Container([
                lay.ren_div,
                lay.res_div,
                lay.rei_div
            ]),
            width={'size': 12, 'offset': 0, 'order': 0}
        ), style = {'textAlign': 'left', 'paddingBottom': '1%'}
    ),
    dbc.Row(        
        dbc.Col(
            dbc.Container([
                dbc.Col(html.Hr()),
                dbc.Col(
                    html.H2("WOP Prediction: {}".format(str(pred)))
                ),
                html.Div(id='my-output')
            ]),
            width={'size': 12, 'offset': 0, 'order': 0}
        ), style = {'textAlign': 'center', 'paddingBottom': '1%'}
    )
])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input('ren-slider', 'value'),
    Input('res-slider', 'value'),
    Input('rei-slider', 'value')
)
def update_prediction(ren, res, rei):
    probe = (
        ('ren', int(ren)), ('rer', int(res)), ('rei', int(rei)),
        ('pct', 1), ('pmd', 1),
        ('mfr', 0), ('mtf', 1), ('fvb', 0)
    )
    vct = np.array([[i[1] for i in probe]])
    # (prediction, bias, contributions) = ti.predict(RF, vct)
    # pred = int(prediction[0][0])
    (prediction, bias, contributions) = (RF.predict(vct), None, None)
    pred = prediction[0]
    return pred


###############################################################################
# Run Dash App
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=False)
