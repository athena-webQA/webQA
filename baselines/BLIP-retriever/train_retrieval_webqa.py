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
from data.webqa_dataset import webqa_dataset

from models.blip_retrieval_webqa import blip_retrieval_webqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

from tqdm.auto import tqdm

import pdb

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        batch['alpha'] = alpha
        batch['device'] = device
        loss_ita = model(**batch)
        loss = loss_ita
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def compute_recallk(scores_text, scores_image, text_qrels, image_qrels, k_list=[5]):
    # TODO: find a better implementation of this

    if len(k_list) > 1:
        raise NotImplementedError("need to implement for k>1")

    k_max = max(k_list)

    text_retrieved = torch.topk(scores_text, k_max, dim=1)
    image_retrieved = torch.topk(scores_image, k_max, dim=1)

    text_retrieved_indices, text_retrieved_values = text_retrieved.indices, text_retrieved.values # Q x K
    image_retrieved_indices, image_retrieved_values = image_retrieved.indices, image_retrieved.values # Q x K

    present = 0
    absent = 0

    present_exact_match = 0
    absent_exact_match = 0

    for query_idx in range(scores_text.shape[0]):

        retrieved_data = [(idx, score, 'text') for idx, score in zip(text_retrieved_indices[query_idx], text_retrieved_values[query_idx])]
        retrieved_data = retrieved_data + [(idx, score, 'image') for idx, score in zip(image_retrieved_indices[query_idx], image_retrieved_values[query_idx])]

        retrieved_results = sorted(retrieved_data, key= lambda x: x[1], reverse=True)[:k_max]

        text_data = set([i[0].int().item() for i in retrieved_results if i[2] == "text"])
        image_data = set([i[0].int().item() for i in retrieved_results if i[2] == "image"])

        text_positives = set(doc_id for doc_id, score in text_qrels[query_idx].items() if score == 1)
        image_positives = set(doc_id for doc_id, score in image_qrels[query_idx].items() if score == 1)

        if len(text_positives) > 0:
            present += len(text_positives.intersection(text_data))
            absent += len(text_positives) - len(text_positives.intersection(text_data))

            res = int((len(text_positives.intersection(text_data))) == len(text_positives) and (len(text_positives) > 0))
            present_exact_match += res
            absent_exact_match += 1 - res

        if len(image_positives) > 0:
            present += len(image_positives.intersection(image_data))
            absent  += len(image_positives) - len(image_positives.intersection(image_data))
        
            res = int(len(image_positives.intersection(image_data)) == len(image_positives) and len(image_positives) > 0)
            present_exact_match += res
            absent_exact_match += 1 - res
        
    return present/(present+absent), present_exact_match/(present_exact_match + absent_exact_match)

@torch.no_grad()
def evaluation(model, question_loader, text_loader, image_loader, text_qrels, image_qrels, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    question_embeds = []
    for idx, batch in tqdm(enumerate(question_loader), total=len(question_loader)):
        question_batch = batch['question']
        question_batch_input = model.tokenizer(question_batch, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        question_batch_output = model.text_encoder(question_batch_input.input_ids, attention_mask = question_batch_input.attention_mask, mode='text')  
        question_batch_embed = F.normalize(model.text_proj(question_batch_output.last_hidden_state[:,0,:]))
        question_embeds.append(question_batch_embed)


    text_embeds = []
    for idx, batch in tqdm(enumerate(text_loader), total=len(text_loader)):
        text_batch = batch['text']
        text_batch_input = model.tokenizer(text_batch, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_batch_output = model.text_encoder(text_batch_input.input_ids, attention_mask = text_batch_input.attention_mask, mode='text')  
        text_batch_embed = F.normalize(model.text_proj(text_batch_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_batch_embed)

    image_embeds = []
    for idx, batch in tqdm(enumerate(image_loader), total=len(image_loader)): 
        
        images = batch['image'].to(device)
        captions = batch['caption']

        caption_inputs = model.tokenizer(captions, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
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
        image_embeds.append(image_outputs)

    question_embeds = torch.cat(question_embeds,dim=0)
    text_embeds = torch.cat(text_embeds,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)

    sim_text = torch.matmul(question_embeds, text_embeds.transpose(0,1)) # Q x T
    sim_image = torch.matmul(question_embeds, image_embeds.transpose(0,1)) # Q x I

    recall_k, recall_exact_match = compute_recallk(sim_text, sim_image, text_qrels, image_qrels)
    print("recall_k, recall_exact_match: ", recall_k, recall_exact_match)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    del question_embeds, text_embeds, image_embeds, sim_text, sim_image

    print('Evaluation time {}'.format(total_time_str)) 
    return recall_k, recall_exact_match

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset = create_dataset('retrieval_%s'%config['dataset'], config, args=args)  
    
    val_question_dataset = val_dataset.question_dataset
    val_image_dataset = val_dataset.image_dataset
    val_text_dataset = val_dataset.text_dataset

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None, None]
    else:
        samplers = [None, None, None, None]
    
    train_loader, val_question_loader, val_image_loader, val_text_loader = create_loader([train_dataset, val_question_dataset, val_image_dataset, val_text_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*4,
                                                          num_workers=[1,1,1,1],
                                                          is_trains=[True, False, False, False], 
                                                          collate_fns=[None, None, None, None])   
    # train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
    #                                                       batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
    #                                                       num_workers=[4,4,4],
    #                                                       is_trains=[True, False, False], 
    #                                                       collate_fns=[None,None,None])   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval_webqa(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model._set_static_graph()
        model_without_ddp = model.module

    ddp_logging_data = model._get_ddp_logging_data()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    for epoch in range(0, config['max_epoch']):    
        if args.evaluate: 
            break

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        train_stats = train(model, train_loader, optimizer, epoch, device, config)  
    
        if utils.is_main_process():  
            recall_k, recall_k_exact_match = evaluation(model_without_ddp, val_question_loader, val_text_loader, val_image_loader, val_dataset.text_qrels, val_dataset.image_qrels, device, config)
      
            print("recall_k, recall_k_exact_match: ", recall_k, recall_k_exact_match)
                                
            if recall_k >= best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = recall_k
                best_epoch = epoch  
                
            val_result = {'recall_k': recall_k, 'recall_k_em': recall_k_exact_match}
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()}                 
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   

            dist.barrier()
        else:
            dist.barrier()

        torch.cuda.empty_cache()
        dist.barrier()

    if utils.is_main_process():  
            recall_k, recall_k_exact_match = evaluation(model_without_ddp, val_question_loader, val_text_loader, val_image_loader, val_dataset.text_qrels, val_dataset.image_qrels, device, config)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_image_only', action='store_true')
    parser.add_argument('--eval_text_only', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)