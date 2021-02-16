import os
import os.path as osp
import pickle
import importlib
import torch
from pantomime_dataset import PantomimeDataset
from torch_geometric.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import sys
import calendar
import time
from models.STN3d import STN3d, count_STN3d
from torch_geometric.nn import DynamicEdgeConv
from models.modified_edgecnn import Net
from temporal_edgecnn.temporal_edgecnn import GeneralizedTemporalSelfAttentionDynamicEdgeConv, count_Multi_head_self_attention,\
    count_GeneralizedTemporalSelfAttentionDynamicEdgeConv, count_DynamicEdgeConv
from torch_multi_head_attention import MultiHeadAttention
from utils.profile_flops import profile
from thop import clever_format

BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='modified_edgecnn',
                    help='Model to run on the data (stgcnn, dgcnn, tgcnn, modified_edgecnn) [default: modified_edgecnn]')
parser.add_argument('--log_dir', default='logs/flops_dgcnn',
                    help='Log dir [default: log]')
parser.add_argument('--k', default=32, type=int, help='Number of nearest points [default: 4]')
parser.add_argument('--t', default=1000, type=int, help='Number of future frames to look at [default: 1]')
parser.add_argument('--spatio_temporal_factor', default=0.01, type=float, help='Spatio-temporal factor [default: 0.01]')
parser.add_argument('--graph_convolution_layers', default=1, type=int, help='Number of graph convolution layers [default: 21]')
parser.add_argument('--gpu_id', default=0, help='GPU ID [default: 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size [default: 32]')
parser.add_argument('--dataset', default='data/pantomime', help='Dataset path. [default: data/pantomime]')
parser.add_argument('--eval_score_path', type=str, default=None, help='Eval score path. [default: dataset path]')
parser.add_argument('--num_class', type=int, default=21, help='Number of classes. [default: 21]')

FLAGS = parser.parse_args()
DATASET = FLAGS.dataset
LOG_DIR = FLAGS.log_dir
K = FLAGS.k
T = FLAGS.t
SPATIO_TEMPORAL_FACTOR = FLAGS.spatio_temporal_factor
GRAPH_CONVOLUTION_LAYERS = FLAGS.graph_convolution_layers
MODEL = importlib.import_module(FLAGS.model)
GPU_ID = FLAGS.gpu_id
BATCH_SIZE = FLAGS.batch_size
EVAL_SCORE_PATH = FLAGS.eval_score_path
NUM_CLASSES = FLAGS.num_class
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

if EVAL_SCORE_PATH is None:
    EVAL_SCORE_PATH = DATASET

model_path = osp.join(LOG_DIR, 'model.pth')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
    sys.stdout.flush()


device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), DATASET)
test_dataset = PantomimeDataset(path, False)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

log_string('Selected model: {}'.format(MODEL))
model = MODEL.Net(NUM_CLASSES, graph_convolution_layers=GRAPH_CONVOLUTION_LAYERS, k=K, T=T, spatio_temporal_factor=SPATIO_TEMPORAL_FACTOR).to(device)
checkpoint = torch.load(Path(model_path))
model.load_state_dict(checkpoint)
log_string('The checkpoint was load successfully: {}'.format(model_path))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
log_string('The number of trainable parameters of the model is {}'.format(params))

model.eval()
count_flops = True
correct = 0
total_y_pred = []
total_y_true = []
total_y_score = None
for data in test_loader:
    data = data.to(device)
    if count_flops:
        macs, params = profile(model, custom_ops={STN3d: count_STN3d,
                                                  MultiHeadAttention: count_Multi_head_self_attention,
                                                  GeneralizedTemporalSelfAttentionDynamicEdgeConv:
                                                  count_GeneralizedTemporalSelfAttentionDynamicEdgeConv,
                                                  DynamicEdgeConv:
                                                  count_DynamicEdgeConv}, inputs=(data,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)
        count_flops = False

    with torch.no_grad():
        batch_pred = model(data)
        if total_y_score is None:
            total_y_score = batch_pred.cpu().detach().numpy()
        else:
            total_y_score = np.concatenate((total_y_score, batch_pred.cpu().detach().numpy()), axis=0)
        pred = batch_pred.max(dim=1)[1]
    correct += pred.eq(data.y.squeeze()).sum().item()
    total_y_pred.extend(pred.cpu().detach().numpy().tolist())
    total_y_true.extend(data.y.squeeze().cpu().detach().numpy().tolist())

with open('{}/scores_{}_{}.pkl'.format(EVAL_SCORE_PATH, FLAGS.model, calendar.timegm(time.gmtime())), 'wb') as handle:
    pickle.dump({
        'total_y_true': total_y_true,
        'total_y_pred': total_y_pred,
        'total_y_score': total_y_score
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)

log_string('Scores have been saved to the dataset directory!')
