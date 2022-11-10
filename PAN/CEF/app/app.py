#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
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
    'WOP': aux.loadModel('HLT', '0.1', 'WOP', 'mlp'),
    'CPT': aux.loadModel('HLT', '0.1', 'CPT', 'mlp')
}
###############################################################################
# Setup Dash App
###############################################################################
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.YETI, dbc_css])
server = app.server
port = int(os.environ.get("PORT", 5000))
app.title = 'pgSIT Cost Effectiveness'

###############################################################################
# Generate Layout
###############################################################################
app.layout = html.Div([
    html.H2(
        "pgSIT Explorer [Prototype]", 
        style={
            'backgroundColor': '#3d348b',
            'color': '#ffffff',
            'textAlign':'center',
            'paddingBottom':'1%', 'paddingTop':'1%'
        }
        # className="bg-primary text-white p-2 mb-2 text-center"
    ),
    dbc.Row(
        dbc.Col(
            # html.Div([ 
            #     html.P("This tool is built for exploration purposes only! For accurate results use MGDrivE!"),
            #     dbc.Col(html.Hr())
            # ])
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
                    html.H4("Window of Protection", style={'textAlign':'center'}),
                    html.H6("(R²: 0.92, MAE: 0.05, RMSE: 0.12)", style={'textAlign':'center', 'font-size': '10px'})
                ]),
                dbc.Row([
                    dbc.Col(html.Div(lay.wop_gauge)),
                    # dbc.Col(html.Div(lay.tti_gauge)), 
                ]),
                dbc.Row([
                    html.H4("Cumulative Potential for Transmission", style={'textAlign':'center'}),
                    html.H6("(R²: 0.94, MAE: 0.04, RMSE: 0.11)", style={'textAlign':'center', 'font-size': '10px'})
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
                # html.Hr(),
                # html.Img(src=app.get_asset_url('SAML.png'), style={'width':'100%'}),
                html.A(
                    "by Héctor M. Sánchez C.", href='https://chipdelmal.github.io/', 
                    target="_blank",
                    style={'color': '#a2d2ff', 'font-size': '15px'}
                )
            ]), 
            style={
                'textAlign':'right',
                'paddingBottom':'0%', 'paddingTop':'0%',
                'paddingLeft': '2%', 'paddingRight': '2%'
            }
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
    return (wop*cst.SIM_TIME/30, 100-cpt*100)


###############################################################################
# Run Dash App
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
