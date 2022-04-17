from collections import defaultdict
import json
import argparse
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file")
    parser.add_argument("--output_file")

    args = parser.parse_args()
    docs = {}
    with open(args.dataset_file, 'r') as fi:
        data_dataset = json.load(fi)

    for id, result in data_dataset.items():
         # pdb.set_trace()

        if len(data_dataset[id]["img_posFacts"]) == 0:
            del data_dataset[id]["img_negFacts"]
            data_dataset[id]["img_negFacts"] = []
        elif len(data_dataset[id]["txt_posFacts"]) == 0:
            del data_dataset[id]["txt_negFacts"]
            data_dataset[id]["txt_negFacts"] = []
    out_file = open(args.output_file, 'w')    
    json.dump(data_dataset,out_file, indent=2)