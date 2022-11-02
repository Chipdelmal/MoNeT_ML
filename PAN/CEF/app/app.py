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


RF = {
    'WOP': aux.loadModel('HLT', '0.1', 'WOP'),
    'TTI': aux.loadModel('HLT', '0.1', 'TTI'),
    'CPT': aux.loadModel('HLT', '0.1', 'CPT'),
}
###############################################################################
# Setup Dash App
###############################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = 'pgSIT Cost Effectiveness'

###############################################################################
# Generate Layout
###############################################################################
app.layout = html.Div([
    dbc.Row(
        dbc.Col(
            html.Div([ 
                html.H2("psSIT Cost Effectiveness"),
                dbc.Col(html.Hr())
            ])
        ), style={
            'textAlign':'center',
            'paddingBottom':'1%', 'paddingTop':'2%'
        }
    ),
    dbc.Row([
        dbc.Col(
            html.Div([
                lay.ren_div, lay.res_div, lay.rei_div,
                dbc.Col(html.Hr()),
                lay.pct_div, lay.pmd_div,
                dbc.Col(html.Hr()),
                lay.mtf_div, 
                dbc.Col(html.Hr()),
                lay.fvb_div, lay.mfr_div
            ]), 
            width=7,
            style={'margin-left':'10px'}
        ),
        dbc.Col(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(html.Div(lay.wop_gauge)), 
                        dbc.Col(html.Div(lay.cpt_gauge))
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(lay.tti_gauge)),
                        dbc.Col(html.Div(lay.tto_gauge))
                    ]
                )
            ]), 
            width=3,
            style={'margin-left':'3px'}
        )
    ])
])

###############################################################################
# Callbacks
###############################################################################
@app.callback(
    Output('wop-gauge', 'value'),
    Output('tti-gauge', 'value'),
    Output('tto-gauge', 'value'),
    Output('cpt-gauge', 'value'),
    Input('ren-slider', 'value'),
    Input('res-slider', 'value'),
    Input('rei-slider', 'value'),
    Input('pct-slider', 'value'),
    Input('pmd-slider', 'value'),
    Input('mfr-slider', 'value'),
    Input('mtf-slider', 'value'),
    Input('fvb-slider', 'value')
)
def update_prediction(ren, res, rei, pct, pmd, mfr, mtf, fvb):
    probe = (
        ('ren', int(ren)), ('rer', int(res)), ('rei', int(rei)),
        ('pct', float(pct)), ('pmd', float(pmd)),
        ('mfr', float(mfr)), ('mtf', float(mtf)), ('fvb', float(fvb))
    )
    vct = np.array([[i[1] for i in probe]])
    # Evaluate models --------------------------------------------------------
    (wop, tti, cpt) = (
        int(RF['WOP'].predict(vct)[0]),
        int(RF['TTI'].predict(vct)[0]),
        float(RF['CPT'].predict(vct)[0])
    )
    tto = wop + tti
    return (wop, tti, tto, cpt)


###############################################################################
# Run Dash App
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=False)
