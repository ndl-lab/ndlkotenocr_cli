#!/usr/bin/env python

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import sys
import os
from pathlib import Path
from .utils import auto_run
from typing import List
import joblib
import pandas as pd


def calcfeatures(tmpcoordlist):
    width=0
    height=0
    for index,t1 in enumerate(tmpcoordlist):
        xmin,ymin,xmax,ymax,text=t1
        width=max(width,xmax)
        height=max(height,ymax)
    width=int(width*1.1)
    height=int(height*1.1)
    coordlist=[]
    arealist=[]
    xcenterlist=[]
    ycenterlist=[]
    widthlist=[]
    heightlist=[]
    xmarginset=set()
    ymarginset=set()
    for index,t1 in enumerate(tmpcoordlist):
        xmin1,ymin1,xmax1,ymax1,text1=t1
        verticalflag1=1
        lxmargin,rxmargin,tymargin,bymargin=[2,2,2,2]
        lxcnt,rxcnt,tycnt,bycnt=[0,0,0,0]
        xmflag=True
        ymflag=True
        for index2,t2 in enumerate(tmpcoordlist):
            xmin2,ymin2,xmax2,ymax2,text2=t2
            if t1==t2:
                continue
            if xmin2<xmin1<xmax2 or (xmax1-xmin1)<(ymax1-ymin1):
                xmflag=False
            if ymin2<ymin1<ymax2 or (xmax1-xmin1)>(ymax1-ymin1):
                ymflag=False
            if xmin1>xmax2:
                lxcnt+=1
                lxmargin=min(lxmargin,(xmin1-xmax2)/width)
            if xmax1<xmin2:
                rxcnt+=1
                rxmargin=min(rxmargin,(xmin2-xmax1)/width)
            #自分の上側
            if ymin1>ymax2:
                tycnt+=1
                tymargin=min(tymargin,(ymin1-ymax2)/height)
            if ymax1<ymin2:
                bycnt+=1
                bymargin=min(bymargin,(ymin2-ymax1)/height)
        if xmflag:
            xmarginset.add(xmin1)
        if ymflag:
            ymarginset.add(ymin1)
        areap=((xmax1-xmin1)/width)*((ymax1-ymin1)/height)
        arealist.append(areap)
        xcenterlist.append((xmin1+xmax1)/(width*2))
        ycenterlist.append((ymin1+ymax1)/(height*2))
        widthlist.append((xmax1-xmin1)/width)
        heightlist.append((ymax1-ymin1)/height)
        coordlist.append([index,xmin1/width,ymin1/height,(xmin1+xmax1)/(width*2),
                          (ymin1+ymax1)/(height*2),(xmax1-xmin1)/width,(ymax1-ymin1)/height,areap,
                          lxmargin,rxmargin,tymargin,bymargin,
                          lxcnt,rxcnt,tycnt,bycnt,verticalflag1,text1])
    coorddf = pd.DataFrame(coordlist, columns = ['order', 'xminp', 'yminp','xcp', 'ycp','widthp','heightp','areap',
                                                 "lxmargin","rxmargin","tymargin","bymargin",
                                                 "lxcnt","rxcnt","tycnt","bycnt","verticalflag","text"])
    coorddf["allcnt"]=len(coordlist)
    coorddf["xmargincnt"]=len(xmarginset)
    coorddf["ymargincnt"]=len(ymarginset)
    coorddf["dummywidth"]=width
    coorddf["dummyheight"]=height
    if len(coordlist)!=0:
        coorddf["verticalp"]=1
        coorddf["areamedian"]=statistics.median(arealist)
        coorddf["wmedian"]=statistics.median(widthlist)
        coorddf["hmedian"]=statistics.median(heightlist)
        coorddf["xcentermedian"]=statistics.median(xcenterlist)
        coorddf["ycentermedian"]=statistics.median(ycenterlist)
    else:
        coorddf["verticalp"]=0
        coorddf["areamedian"]=0
        coorddf["wmedian"]=0
        coorddf["hmedian"]=0
        coorddf["xcentermedian"]=0
        coorddf["ycentermedian"]=0
    return coorddf

def inference_reorder(reordermodel,targetdf):
    if targetdf.shape[0]==0:
        return targetdf
    featurelist=['xminp', 'yminp','xcp','ycp','widthp','heightp',"areap","lxmargin","rxmargin","tymargin","bymargin",
        "lxcnt","rxcnt","tycnt","bycnt",
        "verticalflag","verticalp","wmedian","hmedian","xcentermedian","ycentermedian","xmargincnt","ymargincnt"]
    predorder = reordermodel.predict(targetdf[featurelist]).tolist()
    targetdf["pred_order"]=predorder
    resdf=targetdf.sort_values('pred_order', ascending=True).reset_index(drop=True)
    return resdf

def model_loader(checkpoint: str):
    reordermodel=None
    with open(checkpoint, mode="rb") as f:
        reordermodel = joblib.load(f)
    return reordermodel

class ReadingReorder:
    def __init__(self,checkpoint: str):
        print(f'load from checkpoint={checkpoint}')
        self.load(checkpoint)
    def load(self, checkpoint: str):
        self.model =model_loader(checkpoint)

    def predict(self, coordlist):
        targetdf=calcfeatures(coordlist)
        resdf=inference_reorder(self.model, targetdf)
        resultwords = []
        for index, row in resdf.iterrows():
            xmin = int(row["xminp"] * row["dummywidth"])
            xmax = int((row["xminp"] + row["widthp"]) * row["dummywidth"])
            ymin = int(row["yminp"] * row["dummyheight"])
            ymax = int((row["yminp"] + row["heightp"]) * row["dummyheight"])
            word = {}
            word["boundingBox"] = [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]
            word["id"] = index + 1
            word["isVertical"] = "true"
            word["text"] = row["text"]
            word["isTextline"] = "true"
            word["confidence"] = 1
            resultwords.append(word)
        text = "".join(df["text"].tolist())
        return resultwords,text


def run_reading_reorder(coordlist,checkpoint: str = './models/kotenseki_reading_order_model.joblib'):
    reorder = ReadingReorder(checkpoint)
    resultwords,text = reorder.predict(coordlist)
    print(resultwords,text)

class InferencerWithCLI:
    def __init__(self, conf_dict):
        checkpoint = conf_dict['checkpoint_path']
        self.reorder = ReadingReorder(checkpoint)

    def inference_wich_cli(self, coordlist):
        if self.reorder is None:
            print('ERROR: Reading Reorder is not created.')
            return None
        resultwords,text = self.reorder.predict(coordlist)
        return {'json': resultwords, 'text': text}

if __name__ == '__main__':
    auto_run(run_reading_reorder)
