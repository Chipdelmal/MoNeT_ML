#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from treeinterpreter import treeinterpreter as ti
from dash import html
from dash import Dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)
import layouts as lay
import auxiliary as aux
import constants as cst

RF = {
    'WOP': aux.loadModelNew('HLT', '0.1', 'WOP', 'mlp'),
    'CPT': aux.loadModelNew('HLT', '0.1', 'CPT', 'mlp')
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
                html.H2("psSIT Explorer (prototype)"),
                html.P("This tool is built for exploration purposes only! For accurate results use MGDrivE!"),
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
                lay.mtf_div, 
                dbc.Col(html.Hr()),
                lay.fvb_div, lay.mfr_div,
                dbc.Col(html.Hr()),
                lay.pct_div, lay.pmd_div
            ]), 
            width=8,
            style={'margin-left':'20px'}
        ),
        dbc.Col(
            html.Div([
                dbc.Row([
                    html.H5("Window of Protection", style={'textAlign':'center'}),
                    html.H5("(R2: 0.91, MAE: 125, RMSE: 450)", style={'textAlign':'center'})
                ]),
                dbc.Row([
                    dbc.Col(html.Div(lay.wop_gauge)),
                    # dbc.Col(html.Div(lay.tti_gauge)), 
                ]),
                dbc.Row([
                    html.H5("Cumulative Potential for Transmission", style={'textAlign':'center'}),
                    html.H5("(R2: 0.93, MAE: 0.03, RMSE: 0.1)", style={'textAlign':'center'})
                ]),
                dbc.Row([
                    dbc.Col(html.Div(lay.cpt_gauge)),
                    # dbc.Col(html.Div(lay.tto_gauge))
                ])
            ]), 
            width=3,
            style={'margin-left':'3px'}
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Hr(),
                html.Img(src=app.get_asset_url('SAML.png'), style={'width':'100%'})
            ])
        )
    ])
])

###############################################################################
# Callbacks
###############################################################################
@app.callback(
    Output('wop-gauge', 'value'),
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
        ('ren', int(ren)),   ('rer', int(res)),   ('rei', int(rei)),
        ('pct', float(pct)), ('pmd', float(pmd)),
        ('mfr', float(mfr)), ('mtf', float(mtf)), ('fvb', float(fvb))
    )
    vct = np.array([[i[1] for i in probe]])
    # Evaluate models --------------------------------------------------------
    (wop, cpt) = (
        float(RF['WOP'].predict(vct)[0]),
        float(RF['CPT'].predict(vct)[0])
    )
    return (wop*cst.SIM_TIME, 100-cpt*100)


###############################################################################
# Run Dash App
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8050", debug=True)
