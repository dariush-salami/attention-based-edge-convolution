import os.path as osp
from os import makedirs
import importlib
import torch
from pantomime_dataset import PantomimeDataset
from temporal_transformer import TemporalTransformer
from torch_geometric.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import sys
BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))


parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='dgcnn',
                    help='Model to run on the data (stgcnn, dgcnn) [default: stgcnn]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
FLAGS = parser.parse_args()
LOG_DIR = FLAGS.log_dir
MODEL = importlib.import_module(FLAGS.model)

NUM_CLASSES = 21

model_path = osp.join(LOG_DIR, 'model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/pantomime')
transform = TemporalTransformer()
test_dataset = PantomimeDataset(path, False, pre_transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=6)

print('Selected model: {}'.format(MODEL))
model = MODEL.Net(21).to(device)
checkpoint = torch.load(Path(model_path))
model.load_state_dict(checkpoint)
print('The checkpoint was load successfully: {}'.format(model_path))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of trainable parameters of the model is {}'.format(params))

model.eval()
correct = 0
total_pred = []
total_true = []
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        pred = model(data).max(dim=1)[1]
    correct += pred.eq(data.y.squeeze()).sum().item()
    total_pred.extend(pred.cpu().detach().numpy().tolist())
    total_true.extend(data.y.squeeze().cpu().detach().numpy().tolist())

print(correct / len(test_loader.dataset))
