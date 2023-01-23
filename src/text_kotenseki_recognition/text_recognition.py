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
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from transformers import AutoTokenizer,AutoModelForMaskedLM
import re
import shutil

def create_textline(img,coordobj):
    imageslist = []
    coordslist = []
    for rect in coordobj:
        xmin, ymin, xmax, ymax = rect
        xshift = (xmax - xmin) // 10
        image = cv2.rotate(img[ymin:ymax, xmin + xshift:xmax - xshift], cv2.ROTATE_90_COUNTERCLOCKWISE)
        try:
            image.shape
            imageslist.append(image)
            coordslist.append(rect)
        except:
            continue
    return imageslist,coordslist


class TextRecognizer:
    def __init__(self, preprocessorpath: str,tokenizerpath: str,modelpath:str, device: str,batchsize:int):
        print(f'load from preprocessor={preprocessorpath}, tokenizer={tokenizerpath}, model={modelpath}')
        self.device=device
        self.batchsize=batchsize
        self.load(preprocessorpath, tokenizerpath,modelpath)

    def load(self,preprocessorpath: str,tokenizerpath: str,modelpath:str):
        self.ocrprocessor = TrOCRProcessor.from_pretrained(preprocessorpath)
        self.ocrtokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
        self.ocrmodel = VisionEncoderDecoderModel.from_pretrained(modelpath)
        self.ocrmodel.to(self.device)

    def predict(self, imageslist,coordslist):
        result = []
        imagelength=len(imageslist)
        step_num=imagelength//self.batchsize
        if imagelength%self.batchsize!=0:
            step_num+=1
        for step in range(step_num):
            try:
                generated_ids = self.ocrmodel.generate(
                    self.ocrprocessor(imageslist[step*self.batchsize:(step+1)*self.batchsize],
                                      return_tensors="pt").pixel_values.to(self.device, torch.float))
                restextlist = self.ocrtokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for rect, restext in zip(coordslist[step*self.batchsize:(step+1)*self.batchsize], restextlist):
                    xmin, ymin, xmax, ymax = rect
                    result.append([xmin, ymin, xmax, ymax, restext])
            except:
                pass
        return result

def run_text_recognition(img_paths: str = None, output_path: str = "text_recognition.json",
                         preprocessorpath: str = './models/ndl_kotenseki_layout_config.py',
                         tokenizerpath: str = './models/ndl_kotenseki_layout_config.py',
                         modelpath: str = './models/ndl_kotenseki_layout_config.py',
                         batchsize: int = 100,
                         device: str = 'cuda:0'):
    print("dummy")


class InferencerWithCLI:
    def __init__(self, conf_dict):
        preprocessorpath = conf_dict['saved_preprocessor_model']
        tokenizerpath = conf_dict['saved_tokenize_model']
        modelpath = conf_dict['saved_ocr_model']
        device = conf_dict['device']
        batchsize = conf_dict['batch_size']
        self.recognizer = TextRecognizer(preprocessorpath, tokenizerpath,modelpath, device,batchsize)

    def inference_wich_cli(self, img,coordobj):
        # prediction
        if self.recognizer is None:
            print('ERROR: Layout detector is not created.')
            return None
        imglist, coordslist = create_textline(img,coordobj)
        result = self.recognizer.predict(imglist,coordslist)
        return {'json':result}


if __name__ == '__main__':
    auto_run(run_text_recognition)
