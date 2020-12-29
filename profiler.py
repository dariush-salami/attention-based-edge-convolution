import os
import os.path as osp
import importlib
import torch
from pantomime_dataset import PantomimeDataset
from torch_geometric.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import sys
from pypapi import events, papi_high as high
import time


BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='modified_edgecnn',
                    help='Model to run on the data (stgcnn, dgcnn, tgcnn, modified_edgecnn) [default: modified_edgecnn]')
parser.add_argument('--log_dir', default='logs/two_layers_aug_self_attention_temporal_dynamic_edge_cnn_k_5_max_32_f_32_p_without_outlier_removal',
                    help='Log dir [default: log]')
parser.add_argument('--gpu_id', default=0, help='GPU ID [default: 0]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size [default: 32]')
parser.add_argument('--dataset', default='data/primary_32_f_32_p_without_outlier_removal', help='Dataset path. [default: data/pantomime]')
parser.add_argument('--num_class', type=int, default=21, help='Number of classes. [default: 21]')

FLAGS = parser.parse_args()
DATASET = FLAGS.dataset
LOG_DIR = FLAGS.log_dir
MODEL = importlib.import_module(FLAGS.model)
GPU_ID = FLAGS.gpu_id
BATCH_SIZE = FLAGS.batch_size
NUM_CLASSES = FLAGS.num_class
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

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
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

log_string('Selected model: {}'.format(MODEL))
model = MODEL.Net(NUM_CLASSES).to(device)
checkpoint = torch.load(Path(model_path))
model.load_state_dict(checkpoint)
log_string('The checkpoint was load successfully: {}'.format(model_path))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
log_string('The number of trainable parameters of the model is {}'.format(params))

model.eval()
number_of_batches_to_average = 10
current_batch = 0
total_time = 0
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        start_time = time.time()
        pred = model(data)
        end_time = time.time()
        total_time += end_time - start_time
        current_batch += 1
    if current_batch >= number_of_batches_to_average:
        print("--- %s seconds ---" % (total_time / number_of_batches_to_average))
        break
