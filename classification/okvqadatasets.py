  
from typing import Type, Union
from datasets import load_dataset, load_metric, ClassLabel
import random
import pandas as pd
import json
from transformers import AutoTokenizer
import os
from typing import List, Optional
from transformers import RobertaTokenizerFast, AddedToken
from classification.feature_reader import FeatureReader
from torch.utils.data import Dataset as DS

train_set = "att2in_train.csv"
valid_set = "att2in_valid.csv"
info_file = "info.json"

dir_path = os.path.dirname(os.path.abspath(__file__))
train_key = "train"
eval_key = "val"

class Dataset :
    def __init__(self, image_feature_dir, dataroot, dataset_type, max_seq_length, tokenizer):
        path = dir_path + "/" + dataroot + '/'
        self.datasets = load_dataset('csv', data_files={train_key: path + train_set, eval_key : path + valid_set})
        self.configs = None
        with open(path + info_file) as json_file:
            self.configs = json.load(json_file)
        self.dataset_type = dataset_type
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
    def __call__(self):
        self.datasets = self.datasets.map(self.preprocess_function, batched=True)
        return self.datasets


class OKVQADataset(Dataset) :
    def __init__(self, image_feature_dir, dataroot, dataset_type, max_seq_length, tokenizer):
        super().__init__(image_feature_dir, dataroot, dataset_type, max_seq_length, tokenizer)
        path = dir_path + "/" + dataroot + '/'
        self.datasets = load_dataset('csv', data_files={train_key: path + train_set, eval_key : path + valid_set})
        self.configs = None
        with open(path + info_file) as json_file:
            self.configs = json.load(json_file)
        self.dataset_type = dataset_type
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.image_feature_dir = image_feature_dir
        
    def get_image_path(self, image_id) -> str :
        if self.dataset_type == 'train' :
           image_path = f"COCO_train2014_{str(image_id).zfill(12)}.npy"
        else:
            image_path = f"COCO_val2014_{str(image_id).zfill(12)}.npy"

        return image_path  
    
    def read_features(self, image_path) :
        feature_reader = FeatureReader(
                base_path=self.image_feature_dir, 
                depth_first=False, 
                max_features=100)

        feature, info = feature_reader.read(image_path)

        return feature, info

    def preprocess_function(self, examples):
        total_dict = {}
        captions= [sentence for sentence in examples["caption"]]
        tokenized_examples = self.tokenizer(
            captions, 
            max_length=self.max_seq_length, 
            padding=True,
            truncation=True)

        for k, v in tokenized_examples.items() :
            total_dict[k] = v

        img_id = [ind for ind in examples['img_id']]
        features = []
        for path in img_id :
            feature, info = self.read_features(path)
            features.append(feature)
        labels = [[label] for label in examples['label']]

        total_dict['img_features'] = features
        total_dict['labels'] = labels
        
        return total_dict

if __name__ == "__main__" :

    dataroot = "data"
    max_seq_length = 40

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset_type = 'train'
    image_feature_dir = '/home/aimaster/lab_storage/jinyeong/mmf/data/datasets/okvqa/defaults/features/features_fc6/COCO_trainval2014.lmdb'
    okvqa = OKVQADataset(image_feature_dir=image_feature_dir, dataroot=dataroot, dataset_type=dataset_type, 
                        max_seq_length=max_seq_length, tokenizer=tokenizer)
    processed = okvqa()
    print(processed['train']['input_ids'][0])
    print(tokenizer.decode(processed['train']['input_ids'][0]))
    print("hello")