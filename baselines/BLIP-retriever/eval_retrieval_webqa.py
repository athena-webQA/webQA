'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import gc
import argparse
import os
from xml.etree.ElementPath import prepare_descendant
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from data.webqa_dataset import webqa_dataset

from models.blip_retrieval_webqa import blip_retrieval_webqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

from tqdm.auto import tqdm

import pdb

@torch.no_grad()
def compute_recallk(scores_text, scores_image, text_qrels, image_qrels, k_list=[2,3,5,10,100]):
    # TODO: find a better implementation of this

    # if len(k_list) > 1:
    #     raise NotImplementedError("need to implement for k>1")

    retrieved_results = {}
    k_max = max(k_list)

    text_retrieved = torch.topk(scores_text, k_max, dim=1)
    image_retrieved = torch.topk(scores_image, k_max, dim=1)

    text_retrieved_indices, text_retrieved_values = text_retrieved.indices, text_retrieved.values # Q x K
    image_retrieved_indices, image_retrieved_values = image_retrieved.indices, image_retrieved.values # Q x K

    present = 0
    absent = 0

    present_exact_match = 0
    absent_exact_match = 0


    recall_k = {}

    for k in sorted(k_list):
        for query_idx in range(scores_text.shape[0]):
            # pdb.set_trace()

            retrieved_data = [(idx, score, 'text') for idx, score in zip(text_retrieved_indices[query_idx], text_retrieved_values[query_idx])]
            retrieved_data = retrieved_data + [(idx, score, 'image') for idx, score in zip(image_retrieved_indices[query_idx], image_retrieved_values[query_idx])]

            query_retrieved_results = sorted(retrieved_data, key= lambda x: x[1], reverse=True)[:k]
            retrieved_results[query_idx] = query_retrieved_results

            text_data = set([i[0].int().item() for i in query_retrieved_results if i[2] == "text"])
            image_data = set([i[0].int().item() for i in query_retrieved_results if i[2] == "image"])

            text_positives = set(doc_id for doc_id, score in text_qrels[query_idx].items() if score == 1)
            image_positives = set(doc_id for doc_id, score in image_qrels[query_idx].items() if score == 1)

            if len(text_positives) > 0:
                present += len(text_positives.intersection(text_data))
                absent += len(text_positives) - len(text_positives.intersection(text_data))

                res = int((len(text_positives.intersection(text_data))) == len(text_positives))
                present_exact_match += res
                absent_exact_match += 1 - res

            if len(image_positives) > 0:
                present += len(image_positives.intersection(image_data))
                absent  += len(image_positives) - len(image_positives.intersection(image_data))
            
                res = int(len(image_positives.intersection(image_data)) == len(image_positives))
                present_exact_match += res
                absent_exact_match += 1 - res
            
        recall, em_recall = present/(present+absent), present_exact_match/(present_exact_match + absent_exact_match)
        recall_k[k] = (recall, em_recall)
    
    return recall_k, retrieved_results # last retrieved results will have all k_max values

@torch.no_grad()
def evaluation(model, question_loader, text_loader, image_loader, text_qrels, image_qrels, device, config):
    # test
    model.eval() 
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    question_embeds = []
    question_ids = []
    for idx, batch in tqdm(enumerate(question_loader), total=len(question_loader)):
        
        question_ids.extend(batch['id'])

        question_batch = batch['question']
        question_batch_input = model.tokenizer(question_batch, padding='max_length', truncation=True, max_length=256, return_tensors="pt").to(device) 
        question_batch_output = model.text_encoder(question_batch_input.input_ids, attention_mask = question_batch_input.attention_mask, mode='text')  
        question_batch_embed = F.normalize(model.text_proj(question_batch_output.last_hidden_state[:,0,:]))
        question_embeds.append(question_batch_embed)
        
    text_embeds = []
    text_ids = []
    for idx, batch in tqdm(enumerate(text_loader), total=len(text_loader)):
        text_ids.extend(batch['id'])

        text_batch = batch['text']
        text_batch_input = model.tokenizer(text_batch, padding='max_length', truncation=True, max_length=256, return_tensors="pt").to(device) 
        text_batch_output = model.text_encoder(text_batch_input.input_ids, attention_mask = text_batch_input.attention_mask, mode='text')  
        text_batch_embed = F.normalize(model.text_proj(text_batch_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_batch_embed)
        
    image_embeds = []
    image_ids = []

    for idx, batch in tqdm(enumerate(image_loader), total=len(image_loader)): 
        image_ids.extend(batch['id'])

        images = batch['image'].to(device)

        if config["use_image_grounded_text_encoder"]:
            captions = batch['caption']
            caption_inputs = model.tokenizer(captions, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device) 
            caption_inputs.input_ids[:,0] = model.tokenizer.enc_token_id
            batch_image_embeds = model.visual_encoder(images)
            batch_image_embeds_atts = torch.ones(batch_image_embeds.size()[:-1],dtype=torch.long).to(batch_image_embeds.device)

            image_outputs = model.text_encoder(caption_inputs.input_ids,
                                        attention_mask = caption_inputs.attention_mask,
                                        encoder_hidden_states = batch_image_embeds,
                                        encoder_attention_mask = batch_image_embeds_atts,      
                                        return_dict = True,
                                        )
            image_outputs = F.normalize(model.text_proj(image_outputs.last_hidden_state[:,0,:]),dim=-1)
        else:
            batch_image_embeds = model.visual_encoder(images)
            batch_image_embeds = model.vision_proj(batch_image_embeds[:,0,:])            
            image_outputs = F.normalize(batch_image_embeds, dim=-1) 

        image_embeds.append(image_outputs)

    question_embeds = torch.cat(question_embeds,dim=0) # Q x D
    text_embeds = torch.cat(text_embeds,dim=0) # T x D 
    image_embeds = torch.cat(image_embeds,dim=0) # I x D

    sim_text = torch.matmul(question_embeds, text_embeds.t()) # Q x T
    sim_image = torch.matmul(question_embeds, image_embeds.t()) # Q x I

    with open(f"eval_data/{ config['sim_matrix_prefix'] }-qids.tsv", 'w') as fo:
        for qid in question_ids:
            fo.write(f'{qid}\n')
    with open(f"eval_data/{config['sim_matrix_prefix']}-pids.tsv", 'w') as fo:
        for pid in text_ids:
            fo.write(f'{pid}\n')
    with open(f"eval_data/{config['sim_matrix_prefix']}-img_ids.tsv", 'w') as fo:
        for id in image_ids:
            fo.write(f'{id}\n')

    torch.save(sim_text, f"eval_data/{config['sim_matrix_prefix']}-sim_text.pt")
    torch.save(sim_image, f"eval_data/{config['sim_matrix_prefix']}-sim_image.pt")

    recall_k, retrieved_results = compute_recallk(sim_text, sim_image, text_qrels, image_qrels)
    print("recall_k, recall_exact_match: ")
    pprint(recall_k)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    del question_embeds, text_embeds, image_embeds, sim_text, sim_image

    print('Evaluation time {}'.format(total_time_str)) 
    return recall_k, retrieved_results

def main(args, config):    
    device = torch.device(args.device)

    #### Dataset #### 
    print("Creating retrieval dataset")
    _, val_dataset = create_dataset('retrieval_%s'%config['dataset'], config, args=args)  
    
    val_question_dataset = val_dataset.question_dataset
    val_image_dataset = val_dataset.image_dataset
    val_text_dataset = val_dataset.text_dataset

    samplers = [None, None, None]
    
    val_question_loader, val_image_loader, val_text_loader = create_loader([val_question_dataset, val_image_dataset, val_text_dataset],samplers,
                                                          batch_size=[config['batch_size_test']]*3,
                                                          num_workers=[1,1,1],
                                                          is_trains=[False, False, False], 
                                                          collate_fns=[None, None, None])   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval_webqa(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)
    
    model_without_ddp = model
    
    print("Start eval")
    start_time = time.time()    

    recall_k, retrieved_results = evaluation(model_without_ddp, val_question_loader, val_text_loader, val_image_loader, val_dataset.text_qrels, val_dataset.image_qrels, device, config)

    question_ids = val_question_dataset.get_question_ids()
    text_ids = val_text_dataset.get_text_ids()
    image_ids = val_image_dataset.get_image_ids()


    with open(args.retrieved_results_out_file, 'w') as fo:
        out_results = {}
        for qid, retrieved_values in retrieved_results.items():
            qid = question_ids[qid]
            text_retrieved = [[text_ids[i[0].item()],i[1].item()] for i in retrieved_values if i[-1] == 'text']
            image_retrieved = [[image_ids[i[0].item()],i[1].item()] for i in retrieved_values if i[-1] != 'text']

            out_results[qid] = {}
            out_results[qid]['text_results'] = text_retrieved
            out_results[qid]['image_results'] = image_retrieved

        out_line = json.dumps(out_results, indent=2)
        fo.write(out_line)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('eval time {}'.format(total_time_str)) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_webqa_eval_pretrained.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval_set', choices=['all', 'image', 'text'])
    parser.add_argument('--retrieved_results_out_file', default='pretrained_retrieved_results.json')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)