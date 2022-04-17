import base64
from PIL import Image
from io import BytesIO

import os
import pdb
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from data.webqa_dataset import webqa_dataset

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip_retrieval_webqa import blip_retrieval_webqa

transform = transforms.Compose([
        transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

dataset_dir = "/home/adityasv/webqa/WebQA/WebQA_data_first_release"

dataset = json.load(open(os.path.join(dataset_dir, "WebQA_train_val.json"), "r"))

with open(os.path.join(dataset_dir, "imgs.lineidx"), "r") as fp_lineidx:
        lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

val_data = [v for v in dataset.values() if v['split'] == 'val']

results = {}

def parse_image_datapoint(datapoint):
    # keys: ['image_id', 'title', 'caption', 'url', 'imgUrl']
    image_id = datapoint["image_id"]
    caption = datapoint["caption"]
    
    with open(os.path.join(dataset_dir, "imgs.tsv"), "r") as fp:
        fp.seek(lineidx[int(image_id)%10000000])
        img_base64 = fp.readline().strip().split('\t')
        image = Image.open(BytesIO(base64.b64decode(img_base64[1]))).convert('RGB')

    image = transform(image)  

    out= {}
    out["id"] = image_id
    out["image"] = image
    out["caption"] = caption
    return out

def get_image_embedding(image, caption):
    

for datapoint in val_data:
    qid = datapoint['Guid']

    print(datapoint)

    pos_images = [parse_image_datapoint(i) for i in datapoint['img_posFacts']]
    neg_images = [parse_image_datapoint(i) for i in datapoint['img_negFacts']]

    pos_texts = [i["title"] + " " + i["fact"] for i in datapoint['txt_posFacts']]
    neg_texts = [i["title"] + " " + i["fact"] for i in datapoint['txt_negFacts']]

    question = datapoint["Q"]

    pdb.set_trace()


