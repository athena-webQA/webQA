from collections import defaultdict, namedtuple
import os
import json
from typing import List
from io import BytesIO
from PIL import Image
from matplotlib.pyplot import text
from torch.utils.data import Dataset
import base64
from PIL import Image
from tqdm.auto import tqdm

from dataclasses import dataclass
import numpy as np
import pdb

@dataclass
class QuestionExample:
    question_id: str
    question_text: str
    
@dataclass
class ImageExample:
    image_id: int
    caption: dict

@dataclass
class TextExample:
    text_id: str
    passage: dict

@dataclass
class ImageInput:
    question: str
    pos_examples: List[ImageExample]
    neg_image_examples: List[ImageExample]
    # neg_text_examples: List[TextExample]

@dataclass
class TextInput:
    question: str 
    pos_examples: List[TextExample]
    neg_text_examples: List[TextExample]
    # neg_image_examples: List[ImageExample]

class webqa_dataset(Dataset):
    def __init__(self, transform, image_root, dataset_dir, split, prompt=''):
        
        self.transform = transform
        
        self.image_root = image_root
        self.dataset_dir = dataset_dir
        
        self.prompt = prompt
        if self.prompt != "":
            raise NotImplementedError("need to implement this feature!")

        self.image_data = []
        self.text_data = []

        dataset = json.load(open(os.path.join(self.dataset_dir, "WebQA_train_val.json"), "r"))
        with open(os.path.join(self.dataset_dir, "imgs.lineidx"), "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

        for i in tqdm(dataset.values()):
            if i['split'] == split:
                question = i['Q']
            
                if i['Qcate'] == "text":
                    pos_list = []
                    for pos in i['txt_posFacts']:
                        passage_id = pos['snippet_id']
                        title = pos['title']
                        passage = pos['fact']
                        passage_tokens = title + " " + passage
                        pos_list.append(TextExample(passage_id, passage_tokens))

                    # neg_image_list = []
                    # for neg in i['img_negFacts']:
                    #     image_id = neg['image_id']
                    #     title = neg['title']
                    #     caption = neg['caption']
                    #     caption_tokens = title + " " + caption
                    #     neg_image_list.append(ImageExample(image_id, caption_tokens))

                    neg_text_list = []
                    for neg in i['txt_negFacts']:
                        passage_id = neg['snippet_id']
                        title = neg['title']
                        passage = neg['fact']
                        passage_tokens = title + " " + passage
                        neg_text_list.append(TextExample(passage_id, passage_tokens))
                    if len(neg_text_list) == 0:
                        neg_text_list.append(TextExample("", ""))

                    datapoint = TextInput(question, pos_list, neg_text_list)
                    self.text_data.append(datapoint)

                else:
                    pos_list = []
                    for pos in i['img_posFacts']:
                        image_id = pos['image_id']
                        title = pos['title']
                        caption = pos['caption']
                        caption_tokens = title + " " + caption
                        pos_list.append(ImageExample(image_id, caption_tokens))

                    neg_image_list = []
                    for neg in i['img_negFacts']:
                        image_id = neg['image_id']
                        title = neg['title']
                        caption = neg['caption']
                        caption_tokens = title + " " + caption
                        neg_image_list.append(ImageExample(image_id, caption_tokens))

                    # neg_text_list = []
                    # for neg in i['txt_negFacts']:
                    #     title = neg['title']
                    #     passage = neg['fact']
                    #     passage_tokens = title + " " + passage
                    #     neg_text_list.append(TextExample(passage_tokens))
                    # if len(neg_text_list) == 0:
                    #     neg_text_list.append(TextExample(""))

                    datapoint = ImageInput(question, pos_list, neg_image_list) #, neg_text_list)
                    self.image_data.append(datapoint)

        self.txt2img = {i:i for i in range(len(self.image_data))}
        self.txt2txt = {i:i for i in range(len(self.text_data))}

    def __len__(self):
        return min(len(self.image_data), len(self.text_data))
    
    def parse_image_datapoint(self, datapoint):
        question = datapoint.question
        pos = datapoint.pos_examples.pop(0)
        image_neg = datapoint.neg_image_examples.pop(0)
        
        datapoint.pos_examples.append(pos)
        datapoint.neg_image_examples.append(image_neg)

        pos_image_id = pos.image_id
        neg_image_id = image_neg.image_id
        with open(self.image_root, "r") as fp:
            fp.seek(self.lineidx[int(pos_image_id)%10000000])
            img_base64 = fp.readline().strip().split('\t')
            pos_image = Image.open(BytesIO(base64.b64decode(img_base64[1]))).convert('RGB')
            
            fp.seek(self.lineidx[int(neg_image_id)%10000000])
            img_base64 = fp.readline().strip().split('\t')
            neg_image = Image.open(BytesIO(base64.b64decode(img_base64[1]))).convert('RGB')

        pos_image = self.transform(pos_image)  
        neg_image = self.transform(neg_image)

        out= {}
        out["question"] = question
        out["pos_image"] = pos_image
        out["pos_caption"] = pos.caption
        out["neg_image"] = neg_image
        out["neg_caption"] = image_neg.caption
        return out

    def parse_text_datapoint(self, datapoint):
        question = datapoint.question
        pos = datapoint.pos_examples.pop(0)
        text_neg = datapoint.neg_text_examples.pop(0)
        
        datapoint.pos_examples.append(pos)
        datapoint.neg_text_examples.append(text_neg)

        pos_text = pos.passage
        neg_text = text_neg.passage

        out= {}
        out["question"] = question
        out["pos_txt"] = pos_text
        out["neg_text"] = neg_text
        return out

    def get_text_questions(self):
        questions = []
        for datapoint in self.text_data:
            question = datapoint.question
            questions.append(question)
        return questions
    
    def get_image_questions(self):
        questions = []
        for datapoint in self.image_data:
            question = datapoint.question
            questions.append(question)
        return questions

    def __getitem__(self, index):
        image_datapoint = self.image_data[index%len(self.image_data)]
        text_datapoint = self.text_data[index%len(self.text_data)]
        
        image_datapoint = self.parse_image_datapoint(image_datapoint)
        text_datapoint = self.parse_text_datapoint(text_datapoint)

        out = {}
        for k,v in image_datapoint.items():
            out['image_'+k] = v
        for k,v in text_datapoint.items():
            out['text_'+k] = v
        return out

def create_valid_dataset(image_root, dataset_dir, args):
    question_list = []
    image_list = []
    text_list = []

    question_counter = 0 
    text_counter = 0
    image_counter = 0

    text_qrels = defaultdict(dict)
    image_qrels = defaultdict(dict)

    dataset = json.load(open(os.path.join(dataset_dir, "WebQA_train_val.json"), "r"))

    for i in tqdm(dataset.values()):
        if i['split'] == 'val':
            question_text = i['Q']
            question_id = i['Guid']
            
            question = QuestionExample(question_id, question_text)
            question_list.append(question)

            # if i['Qcate'] == "text":
            for pos in i['txt_posFacts']:
                passage_id = pos['snippet_id']
                title = pos['title']
                passage = pos['fact']
                passage_tokens = title + " " + passage
                text_list.append(TextExample(passage_id, passage_tokens))
                if args.eval_set != 'image':
                    text_qrels[question_counter][text_counter] = 1
                text_counter += 1

            for neg in i['txt_negFacts']:
                passage_id = neg['snippet_id']
                title = neg['title']
                passage = neg['fact']
                passage_tokens = title + " " + passage

                text_list.append(TextExample(passage_id, passage_tokens))
                if args.eval_set != 'image':
                    text_qrels[question_counter][text_counter] = 0
                text_counter += 1
                
            # else:
            for pos in i['img_posFacts']:
                image_id = pos['image_id']
                title = pos['title']
                caption = pos['caption']
                caption_tokens = title + " " + caption
                
                image_list.append(ImageExample(image_id, caption_tokens))
                if args.eval_set != 'text':
                    image_qrels[question_counter][image_counter] = 1
                image_counter += 1

            for neg in i['img_negFacts']:
                image_id = neg['image_id']
                title = neg['title']
                caption = neg['caption']
                caption_tokens = title + " " + caption

                image_list.append(ImageExample(image_id, caption_tokens))
                if args.eval_set != 'image':
                    image_qrels[question_counter][image_counter] = 0
                image_counter += 1

            question_counter += 1

    return question_list, image_list, text_list, text_qrels, image_qrels

class webqa_eval_question_dataset(Dataset):
    def __init__(self, question_list, prompt=''):
        
        self.prompt = prompt
        if self.prompt != "":
            raise NotImplementedError("need to implement this feature!")

        self.question_list = question_list

    def get_question_ids(self):
        return [i.question_id for i in self.question_list]

    def __len__(self):
        return len(self.question_list)
    
    def __getitem__(self, index):
        question: QuestionExample = self.question_list[index]
        id = question.question_id
        text = question.question_text
        out = {"id": id, "question": text}

        return out

class webqa_eval_image_dataset(Dataset):
    def __init__(self, transform, image_root, dataset_dir, image_list):
        
        self.transform = transform
        
        self.image_root = image_root
        self.dataset_dir = dataset_dir
        
        self.image_list = image_list

        with open(os.path.join(self.dataset_dir, "imgs.lineidx"), "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        
    def get_image_ids(self,):
        return [i.image_id for i in self.image_list]

    def __len__(self):
        return len(self.image_list)
    
    def parse_image_datapoint(self, datapoint: ImageExample):
        image_id = datapoint.image_id
        caption = datapoint.caption
        
        with open(self.image_root, "r") as fp:
            fp.seek(self.lineidx[int(image_id)%10000000])
            img_base64 = fp.readline().strip().split('\t')
            image = Image.open(BytesIO(base64.b64decode(img_base64[1]))).convert('RGB')

        image = self.transform(image)  

        out= {}
        out["id"] = image_id
        out["image"] = image
        out["caption"] = caption
        return out

    def __getitem__(self, index):
        image_datapoint: ImageExample = self.image_list[index]
        image_datapoint = self.parse_image_datapoint(image_datapoint)
        
        return image_datapoint

class webqa_eval_text_dataset(Dataset):
    def __init__(self, text_list, prompt=''):
        self.text_list = text_list

    def get_text_ids(self, ):
        return [i.text_id for i in self.text_list]


    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text_datapoint: TextExample = self.text_list[index]

        out = {"id": text_datapoint.text_id, "text": text_datapoint.passage}

        return out        

class webqa_eval_dataset:
    def __init__(self, args, transform, image_root, dataset_dir, split='val', prompt=''):
        
        question_list, image_list, text_list, text_qrels, image_qrels = create_valid_dataset(image_root, dataset_dir, args)

        self.question_dataset = webqa_eval_question_dataset(question_list, prompt=prompt)
        self.image_dataset = webqa_eval_image_dataset(transform, image_root, dataset_dir, image_list)
        self.text_dataset = webqa_eval_text_dataset(text_list)

        print(len(self.question_dataset), len(self.image_dataset), len(self.text_dataset))

        self.image_qrels = image_qrels
        self.text_qrels = text_qrels