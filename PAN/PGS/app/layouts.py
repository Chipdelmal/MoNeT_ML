#!/usr/bin/python
# -*- coding: utf-8 -*-

from dash import html
from dash import dcc

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
# Releases Scheme
###############################################################################
rer_div = html.Div([
    html.H5('Release Size (res):'),
    dcc.Slider(
        id='res-slider',
        min=0, max=50,
        step=5, value=30,
        marks=get_marks(0, 50, 1)
    )
])
