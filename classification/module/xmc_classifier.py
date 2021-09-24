
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import transformers
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers.tokenization_roberta_fast import RobertaTokenizerFast
# from img_encoder import resnet101, resnet50
# from sen_encoder import MultitaskModel 
from transformers import RobertaModel, RobertaTokenizer
from classification.att_lib import *

MODEL_TYPE_DICT={
        "cls": transformers.AutoModelForSequenceClassification,
    }
MODEL_NUM_LABEL_DICT={
        "cls": 2,
    }
TASK_TYPE_AND_MODEL_TYPE={
    "TASK1" : "cls",
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}



class Classifier(nn.Module) :
    def __init__(self, **kwargs) :
        super(Classifier, self).__init__() 
        self.word_contrastive = self.config.word_contrastive
        self.img_contrastive = self.config.img_contrastive
        self.sen_contrastive = self.config.sen_contrastive
        self.sen_pretrained = self.config.mtl_pretrained
        self.max_len = self.config.max_seq_length

        # if self.conig.img_encoder :
        #     img_model_name = self.config_img_encoder
        #     if img_model_name == 'resnet101' :
        #         self.img_encoder = resnet101(pretrained=True)
        #     elif img_model_name == 'resent50' :
        #         self.img_encoder = resnet50(pretrained=True)

        self.sen_model_name = self.config.sen_encoder
        self.sen_encoder = RobertaModel.from_pretrained(self.sen_model_name)

        if self.sen_pretrained :
            self.sen_encoder.load_state_dict(torch.load(self.sen_pretrained))
        
        self.img_projector = nn.ModuleList([
            nn.Linear(2048, 1536)
        ])
        self.sen_projector = nn.ModuleList([
            nn.Linear(768, 1536)
        ])
        self.img_transformation = nn.Conv2d(2048, 786, kernel_size=1)
    
    def forward(self, inputs) :
        # we assume the inputs consists of 3 img feature, 
        # sentence feature, and generated sentence featurey
        img_fc, img_att, sen, gt_sen = inputs  
        img_emb = img_fc
        gt_sen_emb = gt_sen
        token_emb, sen_emb = self.sen_encoder(sen) # output[1] = token embdding, output[1] = sentence embedding 

        img_proj = self.img_projector(img_emb)
        sen_proj = self.sen_projector(sen_emb)
        gt_sen_proj = self.sen_projector(gt_sen_emb) # must consider to do the other projector
        out = nn.Linear(1)(torch.cat([sen_proj, gt_sen_proj]))
        out += torch.sum(img_proj * sen_proj, axis=1, keepdim=True)
        if self.sen_contrastive :
            fake_img_loss, fake_img_acc, fake_img_entropy = contrastive_loss(sen_proj, img_proj)
            real_img_loss, real_img_acc, real_img_entropy = contrastive_loss(gt_sen_proj, img_proj)
        
        if self.word_contrastive :
            
            """
            word _ contrastive loss 계산하고
            x_cond.shape = (112, 16, 16, 384)
            채널 늘린후 conv_fn 으로
            x_cond.shape = (112, 16, 16, 768)
            x_cond_reshape.shape = (112, 256, 768) # 256 total_region_size
            word_feat.shape = (56, 17, 768)
            xmc gan 
            batch size  56
            x.shape = (112, 128, 128, 3)
            x_pool.shape = (112, 1536)
            real_feat.shape = (56, 1536)
            fake_feat.shape = (56, 1536)
            
            x_cond.shape = (56, 100, 2048)
            채널 줄이고
            x_cond_reshape = (56, 100, 768)
            word_feat.shape = (112, 40, 768)
            fake_word_feat = (56, 40, 768)
            real_word_feat = (56, 40, 768)

          
            fake_word_loss, fake_word_acc, fake_word_entropy = attn_lib.word_loss(fake_word_feat, x_cond_reshape, max_len)

            real_word_loss, real_word_acc, real_word_entropy = attn_lib.word_loss(real_word_feat, x_cond_reshape, max_len)  
            """
            img_batch = img_att.shape[0]
            img_ft = img_att.shape[-1]
            img_att = img_att.reshape(img_batch, 10, 10, img_ft)
            img_cond = self.img_transformation(img_att)
           
            fake_word_loss, fake_word_acc, fake_word_entropy = word_loss(fake_word_feat, img_cond, self.max_len)
            
            real_word_loss, real_word_acc, real_word_entropy = word_loss(token_emb, img_cond, self.max_len)

        if self.img_contrastive :
            sen_contrastive_loss, sen_contrastive_acc, sen_contrastive_entropy = contrastive_loss(sen_proj, gt_sen_proj)

        statistic_dict = dict(
            fake_img_loss = fake_img_loss, 
            fake_img_acc = fake_img_acc, 
            fake_img_entropy = fake_img_entropy, 
            real_img_loss = real_img_loss, 
            real_img_acc = real_img_acc, 
            real_img_entropy = real_img_entropy, 
            sen_contrastive_loss = sen_contrastive_loss,
            sen_contrastive_acc = sen_contrastive_acc,
            sen_contrastive_entropy = sen_contrastive_entropy,
        )

        return out, statistic_dict

