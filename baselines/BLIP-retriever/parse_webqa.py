from collections import defaultdict
import json
import argparse
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file")
    parser.add_argument("--dataset_file")
    parser.add_argument("--output_file")

    args = parser.parse_args()

    with open(args.input_file, 'r') as fi:
        input_file = json.load(fi)


    docs = {}
    with open(args.dataset_file, 'r') as fi:
        data_dataset = json.load(fi)

    out_dict = defaultdict(dict)
    with open(args.input_file, 'r') as fi:
        in_dataset = json.load(fi)

    lookup_dict = {}

    for id2, result in data_dataset.items():
        for value in result["txt_posFacts"]:
            lookup_dict[value["snippet_id"]] = value
            
        for value in result["txt_negFacts"]:
            lookup_dict[value["snippet_id"]] = value
        
        for value in result["img_posFacts"]:
            lookup_dict[value["image_id"]] = value
            
        for value in result["img_negFacts"]:
            lookup_dict[value["image_id"]] = value

           
        


    
    # with open(args.output_file, 'w') as fo:
    for id, ret_res in in_dataset.items():
        i = data_dataset[id]
        if i['split'] == 'val':
            out_dict[id]['Q'] = i['Q']
            out_dict[id]['A'] = i['A']
            out_dict[id]['topic'] = i['topic']
            out_dict[id]['split'] = i['split']
            out_dict[id]['Qcate'] = i['Qcate']
            out_dict[id]['Guid'] = i['Guid']

            if len(ret_res['text_results']) != 0:
                out_dict[id]['txt_posFacts'] = []
                for txt_ids in ret_res['text_results']:
                    out_dict[id]['txt_posFacts'].append(lookup_dict[txt_ids[0]])
            else:
                out_dict[id]['txt_posFacts'] = []
            out_dict[id]['txt_negFacts'] = []
        
            if len(ret_res['image_results']) != 0:
                out_dict[id]['img_posFacts'] = []
                for img_ids in ret_res['image_results']:
                    out_dict[id]['img_posFacts'].append(lookup_dict[img_ids[0]])
            else:
                out_dict[id]['img_posFacts'] = []
            out_dict[id]['img_negFacts'] = [] 
    out_file = open(args.output_file, 'w')    
    json.dump(out_dict,out_file, indent=2)
