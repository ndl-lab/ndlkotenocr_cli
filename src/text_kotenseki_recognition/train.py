import pandas as pd
import glob
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from albumentations import Compose, ShiftScaleRotate, RGBShift, \
HueSaturationValue,ToGray, RandomBrightnessContrast, RandomGamma,RandomRain
import numpy as np
from transformers import TrOCRProcessor
from transformers import AutoTokenizer,AutoModelForMaskedLM
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AdamW
import datetime

"""
モデルの再学習について

NDL古典籍OCR学習用データセット(https://github.com/ndl-lab/ndl-minhon-ocrdataset)
の「利用方法」に沿って1行データセットを作成していることを前提としています。

下のconfigにある、ONE_LINE_DATASET_DIRECTORYに1行データセットのディレクトリパスを指定します。
フルスクラッチで学習を行いたい場合、
PRETRAIN_MODEL=None
としてください。
当館が提供しているOCRモデルを利用する場合には、コメントアウトを外して、モデルのディレクトリパスを指定してください。

OUTPUT_DIRに、学習したモデルの出力先を指定してください。

作成したモデルをNDL古典籍OCRモデルで利用する際には、
/root/ndlkotenocr_cli/config.ymlの
text_kotenseki_recognitionのsaved_ocr_modelの示すディレクトリパスを書き換えることで差し替え可能です。

"""

##config
ONE_LINE_DATASET_DIRECTORY="/root/honkoku_oneline/"
PRETRAIN_MODEL=None
#PRETRAIN_MODEL="src/text_kotenseki_recognition/models/kotenseki-trocr-honkoku-v3"
OUTPUT_DIR="/root/text_recognize_model"
BATCH_SIZE=24
##
cer_metric = load_metric("cer")


def onelinedataloader():
    inputfpathlist=[]
    textlist=[]
    for inputimgpath in tqdm(glob.glob(os.path.join(ONE_LINE_DATASET_DIRECTORY,"*"))):
        inputtxtpath=inputimgpath.replace(".jpg",".txt")
        with open(inputtxtpath,"r") as f:
            text=f.read()
            inputfpathlist.append(inputimgpath)
            textlist.append(text.replace("\n",""))
    print(len(textlist))

    df=pd.DataFrame({"file_path":inputfpathlist,"text":textlist})
    train_df, test_df = train_test_split(df, test_size=0.1,random_state=777)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df,test_df


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor,tokenizer,augment=False, max_target_length=100):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer
        self.augment=augment
        self.max_target_length = max_target_length
        self.aug = Compose([
                    ShiftScaleRotate(p=0.9, rotate_limit=5,
                        scale_limit=0.05, border_mode=cv2.BORDER_CONSTANT),
                    ToGray(),
                    RandomBrightnessContrast(),
                    RandomGamma(),
                    RGBShift(),
                    HueSaturationValue()
                    ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_path = self.df['file_path'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_path).convert("RGB")
        image = image.rotate(90, expand=True)
        if self.augment:
            image_np = np.array(image)
            augmented = self.aug(image=image_np)
            image = Image.fromarray(augmented['image'])
            #image=np.unsqueeze(transformed)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(text,padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_cer(pred_ids, label_ids):
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer

def main():
    train_df,test_df=onelinedataloader()
    tokenizer=AutoTokenizer.from_pretrained("src/text_kotenseki_recognition/models/decoder-roberta-v3")

    processor = TrOCRProcessor.from_pretrained("src/text_kotenseki_recognition/models/trocr-base-preprocessor")
    train_dataset = IAMDataset(root_dir='',
                               df=train_df,
                               processor=processor,tokenizer=tokenizer,augment=True)
    eval_dataset = IAMDataset(root_dir='',
                               df=test_df,
                               processor=processor,tokenizer=tokenizer,augment=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if PRETRAIN_MODEL!=None:
        model = VisionEncoderDecoderModel.from_pretrained(PRETRAIN_MODEL)
    model.to(device)
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 100
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(300):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs
            for k, v in batch.items():
                batch[k] = v.to(device)
            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))
        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
            cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer

        total_cer = valid_cer / len(eval_dataloader)
        print("Validation CER:", total_cer)
        if total_cer < 1:
            save_pretrained_dir = os.path.join(OUTPUT_DIR,f'kotenseki-trocr-{total_cer}_{epoch}_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), "JST")).strftime("%Y%m%dT%H%M%S")}')
            model.save_pretrained(save_pretrained_dir)

if __name__ == '__main__':
    main()