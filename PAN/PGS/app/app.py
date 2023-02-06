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
    'WOP': aux.loadModel('HLT', '0.1', 'WOP', 'krs', QNT=None),
    'CPT': aux.loadModel('HLT', '0.1', 'CPT', 'krs', QNT=50),
    'POE': aux.loadModel('HLT', '0.1', 'POE', 'krs', QNT=50)
}
###############################################################################
# Setup Dash App
###############################################################################
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server
port = int(os.environ.get("PORT", 5000))
app.title = 'pgSIT2'

with open('version.txt') as f:
    version = f.readlines()[0]
###############################################################################
# Generate Layout
###############################################################################
app.layout = html.Div([
    html.H2(
        f"pgSITv2 Explorer [Prototype v{version}]", 
        style={
            'backgroundColor': '#a2d2ff',
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
                width=7,
                style={'margin-left':'20px'}
            ),
            dbc.Col(
                html.Div([
                    dbc.Row([
                        html.H4("Reduction on Cumulative Potential for Transmission (CPT)", style={'textAlign':'center', 'font-size': '17.5px'}),
                        # html.H6("(R²: 0.92, MAE: 0.05, RMSE: 0.12)", style={'textAlign':'center', 'font-size': '10px'})
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(lay.cpt_gauge)),
                        # dbc.Col(html.Div(lay.tti_gauge)), 
                    ]),
                    dbc.Row([
                        html.H4("Probability of Elimination (POE)", style={'textAlign':'center', 'font-size': '17.5px'}),
                        # html.H6("(R²: 0.94, MAE: 0.04, RMSE: 0.11)", style={'textAlign':'center', 'font-size': '10px'})
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(lay.poe_gauge)),
                    ])
                ]),
                width=2
            ),
            dbc.Col(
                html.Div([
                    dbc.Row([
                        html.H4("Window of Protection (WOP)", style={'textAlign':'center', 'font-size': '17.5px'}),
                        # html.H6("(R²: 0.94, MAE: 0.04, RMSE: 0.11)", style={'textAlign':'center', 'font-size': '10px'})
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(lay.wop_gauge)),
                    ])
                ]),
                width=2
            )
        ], style={'paddingTop': '1%'}
    ),
    dbc.Row([
        dbc.Col(
            html.Div([
                # html.Hr(),
                html.A(
                    "Disclaimer: This tool was created for exploration purposes only (using entomological-based metrics). For accurate results use",
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
                    "Dev @",
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
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                    html.Hr(),
                    html.H4("Regression Models' PDP/ICE Plots")
                ]
            ), style={'textAlign': 'center'}
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                dbc.Col(html.Div(["", "1-CPT (R² 0.93)", html.Img(src=app.get_asset_url('HLT_50Q_10T_CPT-krs-MLR.png'), style={'width':'90%'})])),
                dbc.Col(html.Div(["", "POE (R² 0.92)", html.Img(src=app.get_asset_url('HLT_50Q_10T_POE-krs-MLR.png'), style={'width':'90%'})])),
                dbc.Col(html.Div(["","WOP (R² 0.88)", html.Img(src=app.get_asset_url('HLT_10T_WOP-krs-MLR.png'), style={'width':'90%'})])),
            ], style={'paddingLeft': '5%'})
        )
    ])
], style={'width': "99.2%"})

###############################################################################
# Callbacks
###############################################################################
@app.callback(
    Output('wop-gauge', 'value'),
    Output('cpt-gauge', 'value'),
    Output('poe-gauge', 'value'),
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
    (wop, cpt, poe) = (
        float(RF['WOP'].predict(vct)[0]),
        float(RF['CPT'].predict(vct)[0]),
        float(RF['POE'].predict(vct)[0])
    )
    if (int(ren)==0 or int(res)==0):
        fMetrics = (0, 0, 0)
    else:
        fMetrics = (
            wop*(cst.SIM_TIME+cst.REL_START)/30, 
            (cpt+cst.REL_START/cst.SIM_TIME)*100, 
            poe*100
        )
    return fMetrics


###############################################################################
# Run Dash App
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
