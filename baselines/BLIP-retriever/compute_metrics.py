import os
import pdb
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", required=True)
args = parser.parse_args()

base_path = "/home/adityasv/webqa/BLIP/eval_data"
prefix = args.prefix

sims_text = torch.load(os.path.join(base_path, f"{prefix}-sim_text.pt"), map_location='cpu')
sims_image = torch.load(os.path.join(base_path, f"{prefix}-sim_image.pt"), map_location='cpu')

text_retrieved = torch.topk(sims_text, 100, dim=1)
image_retrieved = torch.topk(sims_image, 100, dim=1)

text_retrieved_indices, text_retrieved_values = text_retrieved.indices, text_retrieved.values # Q x K
image_retrieved_indices, image_retrieved_values = image_retrieved.indices, image_retrieved.values # Q x K

img_ids = {}
text_ids = {}
qids = {}


with open(os.path.join(base_path, f"{prefix}-img_ids.tsv"), 'r') as fi:
    for idx, line in enumerate(tqdm(fi)):
        img_ids[idx] = line.strip()

with open(os.path.join(base_path, f"{prefix}-pids.tsv"), 'r') as fi:
    for idx, line in enumerate(tqdm(fi)):
        text_ids[idx] = line.strip()

with open(os.path.join(base_path, f"{prefix}-qids.tsv"), 'r') as fi:
    for idx, line in enumerate(tqdm(fi)):
        qids[idx] = line.strip()

reverse_img_ids = {v: k for k,v in img_ids.items()}
reverse_text_ids = {v: k for k,v in text_ids.items()}
reverse_qids = {v:k for k,v in qids.items()}

with open('/home/adityasv/webqa/WebQA/WebQA_data_first_release/WebQA_train_val.json', 'r') as fi:
    dataset = json.load(fi)

val_dataset = {k:v for k,v in dataset.items() if v['split'] == 'val'}



k_list = [2,3,5,10,100]
recall = np.array([0, 0, 0, 0, 0])
recall_distractors = np.array([0, 0, 0, 0, 0])

fret = open('retriever_results.tsv', 'w')

for k_id, k in enumerate(sorted(k_list)):
    present = absent = 0
    present_local = absent_local = 0
    for qix in tqdm(range(len(qids))):
        qid = qids[qix]

        image_positives = []
        image_negatives = []
        for i in val_dataset[qid]['img_posFacts']:
            image_positives.append(reverse_img_ids[str(i['image_id'])])
        for i in val_dataset[qid]['img_negFacts']:
            image_negatives.append(reverse_img_ids[str(i['image_id'])])

        text_positives = []
        text_negatives = []
        for i in val_dataset[qid]['txt_posFacts']:
            text_positives.append(reverse_text_ids[i['snippet_id']])
        for i in val_dataset[qid]['txt_negFacts']:
            text_negatives.append(reverse_text_ids[i['snippet_id']])

        #################################
        pdb.set_trace()
        retrieved_data = [(idx, score, 'text') for idx, score in zip(text_retrieved_indices[qix], text_retrieved_values[qix])]
        retrieved_data = retrieved_data + [(idx, score, 'image') for idx, score in zip(image_retrieved_indices[qix], image_retrieved_values[qix])]

        retrieved_data = sorted(retrieved_data, key= lambda x: x[1], reverse=True)
        query_retrieved_results = retrieved_data[:k]

        text_data = set([i[0].int().item() for i in query_retrieved_results if i[2] == "text"])
        image_data = set([i[0].int().item() for i in query_retrieved_results if i[2] == "image"])

        for i in text_positives:
            if i in text_data:
                present += 1
            else:
                absent += 1
        
        for i in image_positives:
            if i in image_data:
                present += 1
            else:
                absent += 1
        
        #################################

        image_positive_scores = [(i, sims_image[qix][i], "image") for i in image_positives]
        image_negative_scores = [(i, sims_image[qix][i], "image") for i in image_negatives]
        
        text_positive_scores = [(i, sims_text[qix][i], "text") for i in text_positives]
        text_negative_scores = [(i, sims_text[qix][i], "text") for i in text_negatives]

        retrieved_documents = sorted( image_positive_scores + image_negative_scores + text_positive_scores + text_negative_scores, key=lambda x: x[1], reverse=True )
        retrieved_documents = retrieved_documents[:k]

        text_data = set([i[0] for i in retrieved_documents if i[2] == "text"])
        image_data = set([i[0] for i in retrieved_documents if i[2] == "image"])
        
        for i in image_positives:
            if i in image_data:
                present_local += 1
            else:
                absent_local += 1

    recall[k_id] = present/(present+absent)
    recall_distractors[k_id] = present_local/(present_local + absent_local)

print(recall)
print(recall_distractors)