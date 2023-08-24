#!/usr/bin/env python

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import sys
import os
import numpy as np
from reading_order.xy_cut.block_xy_cut import solve

def check_iou(a, b):
    """
    a: [xmin, ymin, xmax, ymax]
    b: [xmin, ymin, xmax, ymax]

    return: array(iou)
    """
    b = np.asarray(b)
    a_area = (a[  2] - a[  0]) * (a[  3] - a[  1])
    b_area = (b[  2] - b[  0]) * (b[  3] - b[  1])
    intersection_xmin = np.maximum(a[0], b[0])
    intersection_ymin = np.maximum(a[1], b[1])
    intersection_xmax = np.minimum(a[2], b[2])
    intersection_ymax = np.minimum(a[3], b[3])
    intersection_w = np.maximum(0, intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0, intersection_ymax - intersection_ymin)
    intersection_area = intersection_w * intersection_h
    min_area=min(a_area,b_area)
    #print(intersection_area,min_area)
    #print(intersection_area/min_area)
    if intersection_area/min_area>0.8 or intersection_area/(a_area+b_area-intersection_area)>0.4:
        #print(intersection_area/min_area)
        return True
    return False

def check_dup(aconf,bconf):
    if check_iou(aconf[:4],bconf[:4]):
        if aconf[-1]>=bconf[-1]:#確信度がないので面積で比較する
            return 1
        else:
            return 2
    return 0

def remove_dup(coordlist):
    complines = list()
    skipset=set()
    for eidx1 in range(len(coordlist)):
        if eidx1 in skipset:
            continue
        element1=coordlist[eidx1]
        bbox1=element1[:4]
        conf1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
        bbox1.append(conf1)
        for eidx2 in range(eidx1+1,len(coordlist)):
            if eidx2 in skipset:
                continue
            element2=coordlist[eidx2]
            bbox2=element2[:4]
            conf2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
            bbox2.append(conf2)
            checkdupval=check_dup(bbox1,bbox2)
            if checkdupval==1:#重なりありかつbbox1の方が確信度高い
                skipset.add(eidx2)
            elif checkdupval==2:
                skipset.add(eidx1)
                break
    lines=list()
    for eidx,element in enumerate(coordlist):
        if not eidx in skipset:
            lines.append(element)
    return lines

def reordertext(coordlist,line_width_scale=1.0):
    lines = np.array([[
        int(line[0]),
        int(line[1]),
        int(line[2]),
        int(line[3]),
    ] for line in coordlist])
    ranks = solve(lines, plot_path=None,logger=None, scale=line_width_scale)
    #zip_ranks=zip(ranks,coordlist)
    zip_sort=sorted(zip(ranks,coordlist))
    s_coordlist=[s[1] for s in zip_sort]
    s_coordlist=remove_dup(s_coordlist)
    text="".join([t[4] for t in s_coordlist])
    return s_coordlist,text

class InferencerWithCLI:
    def __init__(self, conf_dict):
        pass
    def inference_wich_cli(self, obj):
        resultwords,text = reordertext(obj["json"])
        return {'json': resultwords, 'text': text}

