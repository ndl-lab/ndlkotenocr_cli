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
import cv2
import mmcv
from mmdet.apis import (inference_detector, init_detector)


"""
def generate_class_colors(class_num):
    import cv2
    import numpy as np
    colors = 255 * np.ones((class_num, 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(0, 179, class_num)
    colors = cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR)[0]
    return colors


def draw_legand(img, origin, classes, colors, ssz: int = 16):
    import cv2
    c_num = len(classes)
    x, y = origin[0], origin[1]
    for c in range(c_num):
        color = colors[c]
        color = (int(color[0]), int(color[1]), int(color[2]))
        text = classes[c]
        img = cv2.rectangle(img, (x, y), (x + ssz - 1, y + ssz - 1), color, -1)
        img = cv2.putText(img, text, (x + ssz, y + ssz), cv2.FONT_HERSHEY_PLAIN,
                          1, (255, 0, 0), 1, cv2.LINE_AA)
        y += ssz
    return img
"""

class LayoutDetector:
    def __init__(self, config: str, checkpoint: str, device: str):
        print(f'load from config={config}, checkpoint={checkpoint}')
        self.load(config, checkpoint, device)
        #cfg = mmcv.Config.fromfile(config)
        #self.classes = cfg.classes
        #self.colors = generate_class_colors(len(self.classes))

    def load(self, config: str, checkpoint: str, device: str):
        self.model = init_detector(config,checkpoint,None,device)

    def predict(self, img_path: str):
        return inference_detector(self.model, img_path)


    """def show(self, img_path: str, result, score_thr: float = 0.3, border: int = 3, show_legand: bool = True):
        import cv2
        img = cv2.imread(img_path)
        for c in range(len(result)):
            color = self.colors[c]
            color = (int(color[0]), int(color[1]), int(color[2]))
            for pred in result[c]:
                if float(pred[4]) < score_thr:
                    continue
                x0, y0 = int(pred[0]), int(pred[1])
                x1, y1 = int(pred[2]), int(pred[3])
                img = cv2.rectangle(img, (x0, y0), (x1, y1), color, border)

        sz = max(img.shape[0], img.shape[1])
        scale = 1024.0 / sz
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

        if show_legand:
            ssz = 16
            c_num = len(self.classes)
            org_width = img.shape[1]
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, 8 * c_num, cv2.BORDER_REPLICATE)
            x = org_width
            y = img.shape[0] - ssz * c_num
            img = draw_legand(img, (x, y),
                              self.classes, self.colors, ssz=ssz)

        return img

    def draw_rects_with_data(self, img, result, score_thr: float = 0.3, border: int = 3, show_legand: bool = True):
        import cv2
        for c in range(len(result)):
            color = self.colors[c]
            color = (int(color[0]), int(color[1]), int(color[2]))
            for pred in result[c]:
                if float(pred[4]) < score_thr:
                    continue
                x0, y0 = int(pred[0]), int(pred[1])
                x1, y1 = int(pred[2]), int(pred[3])
                img = cv2.rectangle(img, (x0, y0), (x1, y1), color, border)

        sz = max(img.shape[0], img.shape[1])
        scale = 1024.0 / sz
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

        if show_legand:
            ssz = 16
            c_num = len(self.classes)
            org_width = img.shape[1]
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, 8 * c_num, cv2.BORDER_REPLICATE)
            x = org_width
            y = img.shape[0] - ssz * c_num
            img = draw_legand(img, (x, y), self.classes, self.colors, ssz=ssz)

        return img"""


def convert_to_coordlist(result, score_thr: float = 0.3):
    resultlist=[]
    for rect,score in zip(result.pred_instances.bboxes,result.pred_instances.scores):
        if score >= score_thr:
            resultlist.append([int(x) for x in rect.tolist()])
    return resultlist



def run_layout_detection(img_paths: List[str] = None, list_path: str = None, output_path: str = "layout_prediction.json",
                         config: str = './models/ndl_kotenseki_layout_config.py',
                         checkpoint: str = './models/ndl_kotenseki_layout_v1.pth',
                         device: str = 'cuda:0', score_thr: float = 0.3, use_show: bool = False, dump_dir: str = None):
    detector = LayoutDetector(config, checkpoint, device)
    if list_path is not None:
        img_paths = list([s.strip() for s in open(list_path).readlines()])
    if img_paths is None:
        print('Please specify --img_paths or --list_path')
        return -1
    if dump_dir is not None:
        Path(dump_dir).mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        for img_path in img_paths:
            result = detector.predict(img_path)
            resultobj = convert_to_coordlist(result, score_thr=score_thr)
            print(resultobj)
            """if use_show:
                import cv2
                img = detector.show(img_path, result, score_thr=score_thr)
                cv2.namedWindow('show')
                cv2.imshow('show', img)
                if 27 == cv2.waitKey(0):
                    break

            if dump_dir is not None:
                import cv2
                img = detector.show(img_path, result, score_thr=score_thr)
                cv2.imwrite(str(Path(dump_dir) / Path(img_path).name), img)"""


class InferencerWithCLI:
    def __init__(self, conf_dict):
        config = conf_dict['config_path']
        checkpoint = conf_dict['checkpoint_path']
        device = conf_dict['device']
        self.detector = LayoutDetector(config, checkpoint, device)

    def inference_wich_cli(self, img=None, img_path='',
                           score_thr: float = 0.3):
        # prediction
        if self.detector is None:
            print('ERROR: Layout detector is not created.')
            return None
        result = self.detector.predict(img)
        coordobj = convert_to_coordlist(result, score_thr=score_thr)

        """dump_img = None
        if dump is not None:
            dump_img = self.detector.draw_rects_with_data(
                img, result, score_thr=score_thr)"""

        return {'json':coordobj}


if __name__ == '__main__':
    auto_run(run_layout_detection)
