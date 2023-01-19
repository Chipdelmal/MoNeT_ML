#!/usr/bin/python
# -*- coding: utf-8 -*-

PATH_MDL = './models/'

SIM_TIME = 5*365
SA_RANGES = {
    'ren': (0, 52, 2,  30), 
    'rer': (0, 50, 5,  30), 
    'rei': (1, 20, 1,  7),
    'pct': (.5, 1, .05, 0.9), 
    'pmd': (.5, 1, .05, 0.9), 
    'mfr': (0., .5, .05, 0.1), 
    'mtf': (.5, 1, .05, 0.75), 
    'fvb': (0., .5, .05, 0.1)
}