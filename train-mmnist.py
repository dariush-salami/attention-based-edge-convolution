import os
import os.path as osp
from os import makedirs
import importlib
import torch
from mmnist_dataset import MMNISTDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
import sys
from utils.augmentation_transformer import MMNISTTransformer
from chamfer_distance import ChamferDistance

BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='modified_edgecnn_regression',
                    help='Model to run on the data (stgcnn, dgcnn, tgcnn, modified_edgecnn) [default: modified_edgecnn]')
parser.add_argument('--log_dir', default='stgcnn', help='Log dir [default: stgcnn]')
parser.add_argument('--k', default=5, help='Number of nearest points [default: 5]')
parser.add_argument('--t', default=2, help='Number of future frames to look at [default: 5]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 251]')
parser.add_argument('--gpu_id', default=0, help='GPU ID [default: 0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size [default: 32]')
parser.add_argument('--dataset', default='data/mmnist', help='Dataset path. [default: data/pantomime]')
parser.add_argument('--early_stopping', default='True', help='Whether to use early stopping [default: True]')
parser.add_argument('--early_stopping_patience', type=int, default=100,
                    help='Stop the training if there is no improvements after this ' +
                         'number of consequent epochs [default: 100]')

FLAGS = parser.parse_args()
DATASET = FLAGS.dataset
LOG_DIR = FLAGS.log_dir
K = FLAGS.k
T = FLAGS.t
GPU_ID = FLAGS.gpu_id
MAX_EPOCH = FLAGS.max_epoch
MODEL = importlib.import_module(FLAGS.model)
BATCH_SIZE = FLAGS.batch_size
EARLY_STOPPING = FLAGS.early_stopping
EARLY_STOPPING_PATIENCE = FLAGS.early_stopping_patience

if not osp.exists(LOG_DIR):
    print('Creating the model checkpoint directory at {}'.format(LOG_DIR))
    makedirs(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

model_path = osp.join(LOG_DIR, 'model.pth')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
    sys.stdout.flush()


device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), DATASET)
augmentation_transformer = MMNISTTransformer(BATCH_SIZE)
train_dataset = MMNISTDataset(path, True)
test_dataset = MMNISTDataset(path, False)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)#), num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)#, num_workers=6)

model = MODEL.Net(1280*3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = ChamferDistance()

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        data = augmentation_transformer(data)
        optimizer.zero_grad()
        out = model(data).float()
        labels = data.y[:, 1:].float()
        pc_shape = (data.num_graphs, data.num_nodes // data.num_graphs, 3)
        dist1, dist2 = criterion(out.reshape(pc_shape), labels.reshape(pc_shape))
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y.squeeze()).sum().item()
    return correct / len(loader.dataset)


log_string('Selected model: {}'.format(MODEL))
best_acc = -1
best_acc_epoch = -1
current_acc = -1
last_improvement = 0
for epoch in range(1, MAX_EPOCH):
    loss = train()
    current_acc = test(test_loader)
    if current_acc > best_acc:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, loss, current_acc))
        torch.save(model.cpu().state_dict(), model_path)  # saving model
        model.cuda()
        log_string('The model saved in {}'.format(model_path))
        best_acc = current_acc
        best_acc_epoch = epoch
        last_improvement = 0
    elif best_acc > 0:
        log_string(
            'Epoch {:03d}, Train Loss: {:.4f}, Test Accuracy: {}, Best Test Accuracy: {}, Best Epoch: {}'.format(
                epoch, loss, current_acc, best_acc, best_acc_epoch
            ))
        last_improvement += 1
    else:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Test Accuracy: {}'.format(epoch, loss, current_acc))
        last_improvement += 1

    if EARLY_STOPPING == 'True' and last_improvement > EARLY_STOPPING_PATIENCE:
        log_string('No improvement was observed after {} epochs.'.format(last_improvement))
        log_string(
            'The best model with the accuracy of {} on the validation set was saved at epoch {}.'.format(
                best_acc, best_acc_epoch
            ))
        break

if EARLY_STOPPING != 'True':
    log_string('The best model with the accuracy of {} on the validation set was saved at epoch {}.'.format(
        best_acc, best_acc_epoch
    ))
