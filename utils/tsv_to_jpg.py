import json
dataset = json.load(open("../data/WebQA_train_val.json", "r"))

with open("../data/imgs.lineidx", "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
print(len(lineidx))


for k in list(dataset.keys())[:1]:
    for f in dataset[k]['img_posFacts']:
        image_id = f['image_id']
        with open("../data/imgs.tsv", "r") as fp:
            fp.seek(lineidx[int(image_id)%10000000])
            imgid, img_base64 = fp.readline().strip().split('\t')
        #print(image_id, img_base64) # image_id in dataset file and image_id in img file should agree
        im = Image.open(BytesIO(base64.b64decode(img_base64)))
        im.save('../data/gold_features/{}.jpg'.format(image_id))
