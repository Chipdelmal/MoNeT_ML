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
app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server
port = int(os.environ.get("PORT", 5000))
app.title = 'pgSIT Cost Effectiveness'

with open('version.txt') as f:
    version = f.readlines()[0]
###############################################################################
# Generate Layout
###############################################################################
app.layout = html.Div([
    html.H2(
        f"pgSIT Explorer [Prototype v{version}]", 
        style={
            'backgroundColor': '#3d348b',
            'color': '#ffffff',
            'textAlign':'center',
            'paddingBottom':'1%', 'paddingTop':'1%'
        }
    ),
    dbc.Row(
        [
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
                        html.H4("Window of Protection", style={'textAlign':'center', 'font-size': '17.5px'}),
                        html.H6("(R²: 0.92, MAE: 0.05, RMSE: 0.12)", style={'textAlign':'center', 'font-size': '10px'})
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(lay.wop_gauge)),
                        # dbc.Col(html.Div(lay.tti_gauge)), 
                    ]),
                    dbc.Row([
                        html.H4("Reduction on Cumulative Potential for Transmission", style={'textAlign':'center', 'font-size': '17.5px'}),
                        html.H6("(R²: 0.94, MAE: 0.04, RMSE: 0.11)", style={'textAlign':'center', 'font-size': '10px'})
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(lay.cpt_gauge)),
                    ])
                ]), 
                width=3,
                style={'margin-left':'3px', 'margin-right':'0px'}
            )
        ], style={'paddingTop': '1%'}
    ),
    dbc.Row([
        dbc.Col(
            html.Div([
                # html.Hr(),
                # html.Img(src=app.get_asset_url('SAML.png'), style={'width':'100%'}),
                html.A(
                    "Disclaimer: This tool was created for exploration purposes only and using entomological-based metrics only. For accurate results use",
                    style={
                        'color': '#8d99ae', 'font-size': '15px',
                        'textAlign':'left'
                    }
                ),
                html.A(
                    "MGDrivE!",
                    href='https://marshalllab.github.io/MGDrivE/',
                    target="_blank", 
                    style={
                        'color': '#3d348b', 'font-size': '15px',
                        'textAlign':'left', 'paddingLeft': '5px'
                    }
                )
            ]), 
            style={
                'textAlign':'left',
                'paddingBottom':'0%', 'paddingTop':'0%',
                'paddingLeft': '2%', 'paddingRight': '2%'
            }
        ),
        dbc.Col(
            html.Div([
                html.A(
                    "Dev. @",
                    style={'color': '#8d99ae', 'font-size': '15px'}
                ),
                html.A(
                    "Marshall Lab", href='https://www.marshalllab.com/', 
                    target="_blank",
                    style={
                        'color': '#3d348b', 'font-size': '15px',
                        'paddingLeft': '8px'
                    }
                ),
                html.A(
                    "led by",
                    style={
                        'color': '#8d99ae', 'font-size': '15px',
                        'paddingLeft': '8px'
                    }
                ),
                html.A(
                    "Héctor M. Sánchez C.", href='https://chipdelmal.github.io/', 
                    target="_blank",
                    style={
                        'color': '#3d348b', 'font-size': '15px',
                        'paddingLeft': '8px', 'textAlign':'right'
                    }
                )
            ]), 
            style={
                'textAlign': 'right',
                'paddingBottom':'0%', 'paddingTop':'0%',
                'paddingLeft': '2%', 'paddingRight': '2%'
            }
        )
    ])
], style={'width': "99.2%"})

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
