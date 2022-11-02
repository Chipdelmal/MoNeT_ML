#!/usr/bin/python
# -*- coding: utf-8 -*-

from dash import html
from dash import dcc
import constants as cst

def get_marks(start, end, step, norm=None):
    marks = dict()
    for i in range(start, end, step):
        if norm:
            if i == start or i == end - 1:
                marks[int(i / norm)] = str(int(i / norm))
            else:
                marks[i / norm] = str(i / norm)
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
