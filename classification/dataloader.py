import datasets
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error

from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import sys, os

# from transformers.utils.dummy_tokenizers_objects import RobertaTokenizerFast
from transformers import RobertaTokenizerFast 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from random import choices
from collections import Counter

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from classification.okvqadatasets import OKVQADataset

set_verbosity_error()
datasets.logging.set_verbosity(datasets.logging.ERROR)

train_set = "att2in_train.csv"
valid_set = "att2in_valid.csv"
info_file = "info.json"
dataset_path = "./"


def compute_sample_probability(task_configs, split_val, task_ids) :
    probability = []
    total_size = 0
    for task_id in task_ids :
        dataset_size = task_configs[task_id].datasets[split_val].num_rows
        probability.append(dataset_size)
        total_size += dataset_size
    probability[:] = [x / total_size for x in probability]
    
    return probability, total_size


class StrIgnoreDevice(str):
    def to(self, device):

        return self

class OKVQADataLoader:
    def __init__(self, model_name_or_path, batch_size, tokenizer, 
        task_configs, task_datasets, task_datasets_loss, task_datasets_collator, 
        task_datasets_sampler, task_datasets_loader, task_ids, split_val = "train"):

        self.task_configs = task_configs
        self.task_datasets = task_datasets
        self.task_datasets_loss = task_datasets_loss
        self.task_datasets_collator = task_datasets_collator
        self.task_datasets_sampler = task_datasets_sampler
        self.task_datasets_loader = task_datasets_loader
        self.model_name_or_path = model_name_or_path
        self.task_ids = task_ids
        self.tokenizer = tokenizer
        self.split_val = split_val

        self.sampling_probability, self.total_datasize = compute_sample_probability(task_configs, split_val, task_ids)
        print(self.sampling_probability)
        self.batch_size = batch_size
        self.total_steps = int(self.total_datasize / self.batch_size)
        self.cur = 0
        
    @classmethod
    def create(cls, model_name_or_path, task_ids, batch_size, tokenizer, task_args, split="trainval"):
        (task_configs, task_datasets, task_datasets_loss, task_datasets_collator, task_datasets_sampler,
        task_datasets_loader, task_ids ) = cls.LoadDataset(task_ids, batch_size, tokenizer, task_args)

        train_dataset = cls(model_name_or_path, batch_size, tokenizer, task_configs, 
                        task_datasets["train"], task_datasets_loss, task_datasets_collator, 
                        task_datasets_sampler["train"], task_datasets_loader["train"], task_ids, "train")

        eval_dataset = cls(model_name_or_path, batch_size, tokenizer, task_configs, 
                        task_datasets["val"], task_datasets_loss, task_datasets_collator, 
                        task_datasets_sampler["val"], task_datasets_loader["val"], task_ids, "val")
        return train_dataset, eval_dataset

    @classmethod
    def LoadDataset(cls, task_ids, batch_size, tokenizer, task_args, split="trainval"):
        ids = task_ids

        task_configs = {}
        task_datasets = {}
        task_datasets_collator = {}
        task_datasets_sampler = {}
        task_datasets_loader = {}
        task_datasets_loss = {}
        task_ids = []

        task_datasets["train"] = {}
        task_datasets["val"] = {}
     
        task_datasets_sampler["train"] = {}
        task_datasets_sampler["val"] = {}

        task_datasets_loader["train"] = {}
        task_datasets_loader["val"] = {}

        for i, task_id in enumerate(ids) :
            task = 'TASK' + str(task_id)
            task_name = task_args[task]["name"]
            split = task_args[task]["dataset_type"]
            if "train" in split :
                task_ids.append(task)

            img_feature_dir = task_args[task]["img_feature_dir"]
            dataroot = task_args[task]["dataroot"]
            max_seq_length = task_args[task]['max_seq_length']
            loss = task_args[task]['loss']
            task_datasets_loss[task] = loss
            task_configs[task] = OKVQADataset(img_feature_dir, dataroot, split, max_seq_length, tokenizer)
            cur_datasets = task_configs[task]()
            task_datasets_collator[task] = DataCollatorWithPadding(tokenizer)

            if "train" in split :
                task_datasets["train"][task] = cur_datasets["train"]
                task_datasets_sampler["train"][task] = RandomSampler(task_datasets["train"][task])
                task_datasets_loader["train"][task] = DataLoader(
                    task_datasets["train"][task],
                    sampler=task_datasets_sampler["train"][task],
                    collate_fn=task_datasets_collator[task],
                    batch_size=batch_size,
                    pin_memory=True,
                )
            if "val" in split:
                task_datasets["val"][task] = cur_datasets["val"]            
                task_datasets_sampler["val"][task] = RandomSampler(task_datasets["val"][task])
                task_datasets_loader["val"][task] = DataLoader(
                    task_datasets["val"][task],
                    sampler=task_datasets_sampler["val"][task],
                    collate_fn=task_datasets_collator[task],
                    batch_size=batch_size,
                    pin_memory=True,
                )

        return (task_configs, 
            task_datasets,
            task_datasets_loss, 
            task_datasets_collator,
            task_datasets_sampler,
            task_datasets_loader,
            task_ids,
        )
    def _num_each_task_in_batch(self) :
        batch_samples = choices(self.task_ids, self.sampling_probability, k=self.batch_size)

        return Counter(batch_samples)

    def _get_batch(self, task_batch_counter):
        batch = {}
        for task in self.task_ids:
            task_batch = task_batch_counter[task]
            indices = []
            if task_batch == 0 :
                continue
            for i,x in enumerate(self.task_datasets_sampler[task]) : 
                indices.append(x)
                if (i + 1) % task_batch == 0 : 
                    break
            features= [{k:v for k, v in self.task_datasets[task][i].items()} for i in indices]
            print(features)
            batch[task] = self.task_datasets_collator[task](features)
     
        return batch    

    def __iter__(self) :
        self.cur = 0
        while self.total_steps > self.cur:
            task_batch_counter = self._num_each_task_in_batch()
            batch = self._get_batch(task_batch_counter)
            self.cur += 1
            yield batch

    def __len__(self):

        return self.total_steps

    def select(self, total_num):
        self.total_datasize = total_num
        self.total_steps = int(self.total_datasize / self.batch_size)
        if self.split_val == "val":
            for task in self.task_datasets : 
                self.task_datasets[task].select(range(total_num))
                self.task_datasets_loader[task] = DataLoader(
                    self.task_datasets[task],
                    sampler=self.task_datasets_sampler[task],
                    collate_fn=self.task_datasets_collator[task],
                    batch_size=self.batch_size,
                    pin_memory=True,
                )


    def get_task_types(self):

        return self.task_types


def main(args,task_args):
    #task_configs, task_datasets, task_datasets_loss, task_datasets_collator, task_datasets_sampler, task_datasets_loader, task_ids = LoadDataset(args, task_args, ids)
    #sampling_probability = compute_sampling_probability(task_configs, "train")
    #print(sampling_probability)
    #task_batch_counter = num_each_task_in_batch(task_ids, sampling_probability, args["batch_size"])
    #print(task_batch_counter)
    #batch = get_batch(task_ids, task_batch_counter, task_datasets_collator, task_datasets, task_datasets_sampler)
    #for value in batch:
    #    print(batch[value])
    # task 1 multirc: (bn, 511) output: (bn) 0, 1
    # task 2 cola: (bn, 23) output: (bn) 0, 1
    # task 3 socialIQA: (bn, 3, 40) output: (bn) 0, 1, 2
    # task 4 CommonsenseQA: (bn, 5, 28) output: (bn) 0, 1, 2, 3, 4
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # okvqa_loader = OKVQADataLoader(args, task_args)
    okvqa_loader = OKVQADataLoader.create(args["model_name_or_path"], args["task_ids"], args["batch_size"], tokenizer, task_args, split='trainval')
    for i, batch in enumerate(okvqa_loader):
        print("steps : ", i)
        for task_batch in batch:
            print(len(batch[task_batch]['input_ids']))
            # print(task_batch, " : ", len(batch[task_batch]['labels']))

if __name__ == '__main__':
    import yaml
    import os
    from os.path import dirname, abspath
    print(dirname(dirname(abspath(__file__))))
    with open(dirname(dirname(abspath(__file__))) +'/classification' + '/task.yml') as f:
        task_args = yaml.load(f, Loader=yaml.FullLoader)
        print(task_args['TASK1'])
    args = {"model_name_or_path" : "roberta-base", "batch_size" : 1, "task_ids" : [1]}
    main(args, task_args)
