import pdb
from tqdm import tqdm
import json 
import numpy as np
import random

with open('/home/adityasv/webqa/WebQA/WebQA_data_first_release/WebQA_train_val.json', 'r')  as fi:
    dataset = json.load(fi)

train_data = []
val_data = []

for i in dataset.values():
    split = i['split']

    if split == 'train':
        train_data.append(i)
    else:
        val_data.append(i)

with open('/home/adityasv/webqa/seq2seq/webqa_dataset/train.idx', 'w') as fidx:
    with open('/home/adityasv/webqa/seq2seq/webqa_dataset/train.source', 'w') as fsource:
        with open('/home/adityasv/webqa/seq2seq/webqa_dataset/train.target', 'w') as ftarget:
            for example in tqdm(train_data):
                # pdb.set_trace()

                question_id = example['Guid']
                q = example['Q']
                assert len(example['A']) == 1
                a = example['A'][0]
                
                q = q.strip("\"")
                a = a.strip("\"")

                q = q.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()
                a = a.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()

                pos = []
                neg = []
                for i in example['txt_posFacts']:
                    title = i['title']
                    passage = i['fact']
                    ex = title + " " + passage
                    ex = ex.strip().replace("\n", "")
                    pos.append(ex)
                
                for i in example['img_posFacts']:
                    caption = i['caption']
                    caption = caption.replace("\n", "")
                    pos.append(caption)
                
                for i in example['txt_negFacts']:
                    title = i['title']
                    passage = i['fact']
                    ex = title + " " + passage
                    ex = ex.strip().replace("\n", "")
                    neg.append(ex)
                
                for i in example['img_negFacts']:
                    caption = i['caption']
                    caption = caption.replace("\n", "")
                    neg.append(caption)

                num_negs = 0

                # if len(pos) == 5:
                #     print(example['txt_posFacts'])
                #     print(example['img_posFacts'])

                while num_negs <= 0:
                    num_negs = np.random.poisson(10 - len(pos))
                
                num_negs = min(num_negs, len(neg))

                if len(neg) < 2:
                    negs = neg
                else:
                    negs = random.sample(neg, num_negs)

                data = pos + negs
                random.shuffle(data)
                source = q + ' </s> ' + ' </s> '.join([i.strip() for i in data]) + '\n'
                source = source.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()
                
                assert len(source) > 0 and len(a) > 0, f"{source} SEP {a}"

                fidx.write(f"{question_id}\n")
                fsource.write(source + "\n")
                ftarget.write(a.strip() + '\n')

with open('/home/adityasv/webqa/seq2seq/webqa_dataset/val.idx', 'w') as fidx:
    with open('/home/adityasv/webqa/seq2seq/webqa_dataset/val.source', 'w') as fsource:
        with open('/home/adityasv/webqa/seq2seq/webqa_dataset/val.target', 'w') as ftarget:
            for example in tqdm(val_data):
                # pdb.set_trace()

                question_id = example['Guid']
                q = example['Q']
                assert len(example['A']) == 1
                a = example['A'][0]
                
                q = q.strip("\"")
                a = a.strip("\"")

                q = q.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()
                a = a.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()

                pos = []
                neg = []
                for i in example['txt_posFacts']:
                    title = i['title']
                    passage = i['fact']
                    ex = title + " " + passage
                    ex = ex.strip().replace("\n", "")
                    pos.append(ex)
                
                for i in example['img_posFacts']:
                    caption = i['caption']
                    caption = caption.replace("\n", "")
                    pos.append(caption)
                
                for i in example['txt_negFacts']:
                    title = i['title']
                    passage = i['fact']
                    ex = title + " " + passage
                    ex = ex.strip().replace("\n", "")
                    neg.append(ex)
                
                for i in example['img_negFacts']:
                    caption = i['caption']
                    caption = caption.replace("\n", "")
                    neg.append(caption)

                num_negs = 0

                # if len(pos) == 5:
                #     print(example['txt_posFacts'])
                #     print(example['img_posFacts'])

                while num_negs <= 0:
                    num_negs = np.random.poisson(10 - len(pos))
                
                num_negs = min(num_negs, len(neg))

                if len(neg) < 2:
                    negs = neg
                else:
                    negs = random.sample(neg, num_negs)

                data = pos + negs
                random.shuffle(data)
                source = q + ' </s> ' + ' </s> '.join([i.strip() for i in data]) + '\n'
                source = source.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()
                assert len(source) > 0 and len(a) > 0, f"{source} SEP {a}"

                fidx.write(f"{question_id}\n")
                fsource.write(source + "\n")
                ftarget.write(a.strip() + '\n')

with open('/home/adityasv/webqa/seq2seq/webqa_dataset/test.idx', 'w') as fidx:
    with open('/home/adityasv/webqa/seq2seq/webqa_dataset/test.source', 'w') as fsource:
        with open('/home/adityasv/webqa/seq2seq/webqa_dataset/test.target', 'w') as ftarget:
            for example in tqdm(val_data):
                # pdb.set_trace()

                question_id = example['Guid']
                q = example['Q']
                assert len(example['A']) == 1
                a = example['A'][0]
                
                q = q.strip("\"").strip()
                a = a.strip("\"").strip()

                q = q.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()
                a = a.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()

                pos = []
                neg = []
                for i in example['txt_posFacts']:
                    title = i['title']
                    passage = i['fact']
                    ex = title + " " + passage
                    ex = ex.strip().replace("\n", "")
                    pos.append(ex)
                
                for i in example['img_posFacts']:
                    caption = i['caption']
                    caption = caption.replace("\n", "")
                    pos.append(caption)
                
                for i in example['txt_negFacts']:
                    title = i['title']
                    passage = i['fact']
                    ex = title + " " + passage
                    ex = ex.strip().replace("\n", "")
                    neg.append(ex)
                
                for i in example['img_negFacts']:
                    caption = i['caption']
                    caption = caption.replace("\n", "")
                    neg.append(caption)

                num_negs = 0

                # if len(pos) == 5:
                #     print(example['txt_posFacts'])
                #     print(example['img_posFacts'])

                while num_negs <= 0:
                    num_negs = np.random.poisson(10 - len(pos))
                
                num_negs = min(num_negs, len(neg))

                if len(neg) < 2:
                    negs = neg
                else:
                    negs = random.sample(neg, num_negs)

                data = pos + negs
                random.shuffle(data)
                source = q + ' </s> ' + ' </s> '.join([i.strip() for i in data]) + '\n'

                source = source.replace("\n", "").replace("\r", " ").replace("\t", " ").strip()
                assert len(source) > 0 and len(a) > 0, f"{source} SEP {a}"

                fidx.write(f"{question_id}\n")
                fsource.write(source + "\n")
                ftarget.write(a + '\n')