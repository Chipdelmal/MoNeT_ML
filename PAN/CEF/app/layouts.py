#!/usr/bin/python
# -*- coding: utf-8 -*-

from dash import html
from dash import dcc
import dash_daq as daq
import constants as cst

def get_marks(start, end, step, norm=None):
    marks = dict()
    for i in range(start, end, step):
        if norm:
            if i == start or i == end - 1:
                marks[int(i/norm)] = str(int(i/norm))
            else:
                marks[i/norm] = str(i/norm)
        else:
            marks[i] = str(i)
    return marks

def get_marks_float(start, end, step, multiplier=100):
    marks = dict()
    (st, en, st) = [
        int(i*multiplier) for i in (start, end, step)
    ]
    for i in range(st, en, st):
        if i==st or i ==(en-1):
            marks[int(i)] = str(int(i))
        else:
            marks[i] = str(i)
    return marks

###############################################################################
# Releases Scheme Sliders
###############################################################################
(REN, RES, REI) = [
    cst.SA_RANGES[i] for i in ('ren', 'rer', 'rei')
]
ren_div = html.Div([
    html.H5('Number of Releases (ren):'),
    dcc.Slider(
        id='ren-slider',
        min=REN[0], max=REN[1],
        step=REN[-2], value=REN[-1],
        marks=get_marks(REN[0], REN[1], REN[-2])
    )
])
res_div = html.Div([
    html.H5('Release Size (res):'),
    dcc.Slider(
        id='res-slider',
        min=RES[0], max=RES[1],
        step=RES[-2], value=RES[-1],
        marks=get_marks(RES[0], RES[1], RES[-2])
    )
])
rei_div = html.Div([
    html.H5('Release Interval (rei):'),
    dcc.Slider(
        id='rei-slider',
        min=REI[0], max=REI[1],
        step=REI[-2], value=REI[-1],
        marks=get_marks(REI[0], REI[1], REI[-2])
    )
])
###############################################################################
# Construct Sliders
###############################################################################
(PCT, PMD, MFR, MTF, FVB) = [
    cst.SA_RANGES[i] for i in ('pct', 'pmd', 'mfr', 'mtf', 'fvb')
]
pct_div = html.Div([
    html.H5('Cutting Rate (pct):'),
    dcc.Slider(
        id='pct-slider',
        min=PCT[0], max=PCT[1],
        step=PCT[-2], value=PCT[-1],
        marks=get_marks_float(PCT[0], PCT[1], PCT[-2])
    )
])
pmd_div = html.Div([
    html.H5('Maternal Deposition (pmd):'),
    dcc.Slider(
        id='pmd-slider',
        min=PMD[0], max=PMD[1],
        step=PMD[-2], value=PMD[-1],
        marks=get_marks_float(PMD[0], PMD[1], PMD[-2])
    )
])
mfr_div = html.Div([
    html.H5('Male Fertility (mfr):'),
    dcc.Slider(
        id='mfr-slider',
        min=MFR[0], max=MFR[1],
        step=MFR[-2], value=MFR[-1],
        marks=get_marks_float(MFR[0], MFR[1], MFR[-2])
    )
])
mtf_div = html.Div([
    html.H5('Mating Fitness (mtf):'),
    dcc.Slider(
        id='mtf-slider',
        min=MTF[0], max=MTF[1],
        step=MTF[-2], value=MTF[-1],
        marks=get_marks_float(MTF[0], MTF[1], MTF[-2])
    )
])
fvb_div = html.Div([
    html.H5('Female Viability (fvb):'),
    dcc.Slider(
        id='fvb-slider',
        min=FVB[0], max=FVB[1],
        step=FVB[-2], value=FVB[-1],
        marks=get_marks_float(FVB[0], FVB[1], FVB[-2])
    )
])

###############################################################################
# Output Gauges
###############################################################################
wop_gauge = daq.Gauge(
    id='wop-gauge',
    color={
        "gradient": True,
        "ranges": {
            "#FF006E": [0,      3*365], 
            "#ABE2FB": [3*365,  6*365], 
            "#FFFFFF": [6*365, 10*365]
        }
    },
    size=250,
    label=' ',
    value=0, min=0, max=365*10,
    showCurrentValue=True, units="days"
)
tti_gauge = daq.Gauge(
    id='tti-gauge',
    color={
        "gradient": True,
        "ranges": {
            "#FF006E": [0,      3*365], 
            "#ABE2FB": [3*365,  6*365], 
            "#FFFFFF": [6*365, 10*365]
        }
    },
    size=150,
    label=' ',
    value=0, min=0, max=365*10,
    showCurrentValue=False, units="day"
)
tto_gauge = daq.Gauge(
    id='tto-gauge',
    color={
        "gradient": True,
        "ranges": {
            "#FF006E": [0,      3*365], 
            "#ABE2FB": [3*365,  6*365], 
            "#FFFFFF": [6*365, 10*365]
        }
    },
    size=150,
    label=' ',
    value=0, min=0, max=365*10,
    showCurrentValue=False, units="day"
)
cpt_gauge = daq.Gauge(
    id='cpt-gauge',
    color={
        "gradient": True,
        "ranges": {
            "#FFFFFF": [0,   25],
            "#ABE2FB": [25,  50], 
            "#FF006E": [50, 100]
        }
    },
    size=250,
    label=' ',
    value=0, min=0, max=100,
    showCurrentValue=True, units="%AUC"
)


