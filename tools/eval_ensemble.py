from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import redirect_stderr

import json
import numpy as np

import time
import os
from six.moves import cPickle

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils_uncertainty as eval_utils_un
# import captioning.utils.eval_utils as eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--ids', nargs='+', required=True, help='id of the models to ensemble')
parser.add_argument('--weights', nargs='+', required=False, default=None, help='id of the models to ensemble')
parser.add_argument('--models', nargs='+', required=True,
                help='path to model to evaluate')
parser.add_argument('--infos_paths', nargs='+', required=True, help='path to infos to evaluate')
parser.add_argument('--uncertainty_lambda', type=int, default=0, help='adjusting weight of uncertainty for beam_search')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)

opt = parser.parse_args()

model_infos = []
model_paths = []
for model, info_path in zip(opt.models, opt.infos_paths):
    
    model_infos.append(utils.pickle_load(open(model, 'rb')))
    model_paths.append(info_path)

# Load one infos
infos_1 = model_infos[1]

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
for k in replace:
    setattr(opt, k, getattr(opt, k) or getattr(infos_1['opt'], k, ''))

vars(opt).update({k: vars(infos_1['opt'])[k] for k in vars(infos_1['opt']).keys() if k not in vars(opt)}) # copy over options from model


opt.use_box = max([getattr(infos['opt'], 'use_box', 0) for infos in model_infos])
assert max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]), 'Not support different norm_att_feat'
assert max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]), 'Not support different norm_box_feat'

vocab = infos_1['vocab'] # ix -> word mapping
opt.vocab = vocab


# Setup the model
from captioning.models.AttEnsemble_uncertainty import AttEnsemble

_models = []
for i in range(len(model_infos)):
    model_infos[i]['opt'].start_from = None
    model_infos[i]['opt'].vocab = vocab
    tmp = models.setup(model_infos[i]['opt'])
    tmp.load_state_dict(torch.load(model_paths[i]))
    _models.append(tmp)

if opt.weights is not None:
    opt.weights = [float(_) for _ in opt.weights]
model = AttEnsemble(_models, weights=opt.weights)
model.seq_length = opt.max_length
model.vocab = opt.vocab
del opt.vocab
model.cuda()
model.eval()
crit = losses.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos_1['vocab']

opt.id = '+'.join([id + str(weight) for id, weight in zip(opt.ids, opt.weights)])
opt.dataset = opt.input_json

print(opt)

# Set sample options
loss, split_predictions, lang_stats = eval_utils_un.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
