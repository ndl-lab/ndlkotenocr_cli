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
    lines=list()
    complines = list()
    for element in coordlist:
        xmin,ymin,xmax,ymax,text=element
        conf = (xmax-xmin)*(ymax-ymin)
        checkdupval=0
        if len(lines)!=0:
            checkdupval=check_dup(complines[-1],[xmin,ymin,xmax,ymax,conf])
        if checkdupval==0:#重複なし
            lines.append(element)
            complines.append([xmin,ymin,xmax,ymax,conf])
        elif checkdupval==1:#重複あり （今見ているlineをスキップ）
            continue
        elif checkdupval==2:#重複あり（比較対象を削除）
            del lines[-1]
            del complines[-1]
            lines.append(element)
            complines.append([xmin,ymin,xmax,ymax,conf])
        else:
            print("error!")
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

