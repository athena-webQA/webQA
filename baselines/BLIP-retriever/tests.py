import sys

from pyrsistent import b
sys.path.append("/Users/ezio/Downloads/CMU/11777/BLIP/data")

import logging 
logging.basicConfig(level=logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval_webqa import blip_retrieval_webqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

import pdb

if __name__ == "__main__":

    config = yaml.load(open("./configs/retrieval_webqa.yaml", 'r'), Loader=yaml.Loader)
    train_dataset, val_dataset = create_dataset('retrieval_webqa', config)  

    val_question_dataset = val_dataset.question_dataset
    val_image_dataset = val_dataset.image_dataset
    val_text_dataset = val_dataset.text_dataset

    # print(train_dataset[65])

    train_loader = DataLoader(train_dataset, batch_size=2)
    x = next(iter(train_loader))
    # print(x)

    samplers = [None, None, None, None]
    train_loader, val_question_loader, val_image_loader, val_text_loader = create_loader([train_dataset, val_question_dataset, val_image_dataset, val_text_dataset],samplers,
                                                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*4,
                                                            num_workers=[1,1,1,1],
                                                            is_trains=[True, False, False, False], 
                                                            collate_fns=[None, None, None, None])     

    model = blip_retrieval_webqa(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    # pdb.set_trace()
    batch = next(iter(train_loader))
    batch['alpha'] = 0.1
    print(batch.keys())
    outputs = model(**batch)
    print(outputs)