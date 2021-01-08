import os
import os.path as osp
from os import makedirs
import importlib
import torch
from mmnist_dataset import MMNISTDataset
from torch_geometric.data import DataLoader, Data
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from utils.augmentation_transformer import MMNISTTransformer
from chamfer_distance import ChamferDistance
from emd import EarthMoverDistance

BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='modified_edgecnn_regression',
                    help='Model to run on the data (stgcnn, dgcnn, tgcnn, modified_edgecnn) [default: modified_edgecnn]')
parser.add_argument('--log_dir', default='temporal_regression_mmnist_straight', help='Log dir [default: stgcnn]')
parser.add_argument('--k', default=5, help='Number of nearest points [default: 5]')
parser.add_argument('--t', default=2, help='Number of future frames to look at [default: 5]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default=1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 251]')
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
CD_ALPHA = FLAGS.alpha
EMD_BETA = FLAGS.beta

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

def preprocess_validation(valid):
    x = valid[:, :10]
    y = valid[:, 10:]

    x_seq_number = np.expand_dims(np.arange(1, 11), axis=(0, 1, 2))
    y_seq_number = np.expand_dims(np.arange(11, 21), axis=(0, 1, 2))
    x_seq_number = np.repeat(x_seq_number, x.shape[2], axis=-1).reshape(1, 10, -1, 1)
    y_seq_number = np.repeat(y_seq_number, x.shape[2], axis=-1).reshape(1, 10, -1, 1)
    x_seq_number = np.tile(x_seq_number, (x.shape[0], 1, 1, 1))
    y_seq_number = np.tile(y_seq_number, (y.shape[0], 1, 1, 1))

    x = np.concatenate([x_seq_number, x, np.zeros([x.shape[0], x.shape[1], x.shape[2], 1])], axis=-1)
    y = np.concatenate([y_seq_number, y, np.zeros([y.shape[0], y.shape[1], y.shape[2], 1])], axis=-1)

    x = x.reshape(x.shape[0], x.shape[1] * y.shape[2], x.shape[3])
    y = y.reshape(x.shape[0], y.shape[1] * y.shape[2], y.shape[3])

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    return [Data(x=i, y=j) for i, j in zip(x, y)]

device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), DATASET)

train_dataset = MMNISTDataset(path, True)
test_dataset = np.load('test-1mnist-64-128point-20step.npy')
test_dataset = preprocess_validation(test_dataset)
# mmnist test set generation
# augmentation_transformer = MMNISTTransformer(len(test_dataset))
# test_data = augmentation_transformer(test_dataset.data)


augmentation_transformer = MMNISTTransformer(BATCH_SIZE)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

model = MODEL.Net(3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
ch = ChamferDistance()
emd = EarthMoverDistance()





def calc_loss(data, out, labels):
    loss = ch_loss = emd_loss = 0
    pc_shape = (data.num_graphs, 10, data.num_nodes//(data.num_graphs*10), 3)
    out = out.reshape(pc_shape)
    labels = labels.reshape(pc_shape)
    for t in range(10):
        pred_frame = out[:, t, :, :].squeeze()
        frame = labels[:, t, :, :].squeeze()

        dist_forward, dist_backward = ch(pred_frame, frame)
        ch_dist = (torch.mean(dist_forward)) + (torch.mean(dist_backward))
        ch_loss += ch_dist

        emd_dist = torch.mean(emd(pred_frame, frame, transpose=False))
        emd_loss += emd_dist
        loss += (CD_ALPHA*ch_dist) + (EMD_BETA*emd_dist)

    loss /= 10
    ch_loss /= 10
    emd_loss /= (128 * 10)

    return loss, ch_loss, emd_loss


def train():
    model.train()

    total_loss = 0
    total_ch_loss = 0
    total_em_loss = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        data = augmentation_transformer(data)
        optimizer.zero_grad()
        out = model(data).float()
        labels = data.y[:, 1:].float()
        loss, ch_loss, emd_loss = calc_loss(data, out, labels)
        loss.backward()
        
        total_loss += loss.item() * data.num_graphs
        total_ch_loss += ch_loss.item() * data.num_graphs
        total_em_loss += emd_loss.item() * data.num_graphs
        
        optimizer.step()
        step += 1
    return total_loss / len(train_dataset), total_ch_loss / len(train_dataset), total_em_loss / len(train_dataset)


def test(loader):
    model.eval()
    total_loss = 0
    total_ch_loss = 0
    total_em_loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            labels = data.y[:, 1:]
            loss, ch_loss, emd_loss = calc_loss(data, out, labels)
            total_loss += loss.item() * data.num_graphs
            total_ch_loss += ch_loss.item() * data.num_graphs
            total_em_loss += emd_loss.item() * data.num_graphs 
            
    return total_loss / len(train_dataset), total_ch_loss / len(train_dataset), total_em_loss / len(train_dataset)


log_string('Selected model: {}'.format(MODEL))

best_loss = 1e8
best_ch_loss = 1e8
best_emd_loss = 1e8
best_loss_epoch = -1
current_loss = 0
current_emd_loss = 0
current_ch_loss = 0
last_improvement = 0

for epoch in range(1, MAX_EPOCH):
#     current_loss, current_ch_loss, current_emd_loss = test(test_loader)
    loss,_,_ = train()
    scheduler.step()
    current_loss, current_ch_loss, current_emd_loss = test(test_loader)
    if current_loss < best_loss:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Chamfer: {:.4f}, Test EMD: {:.4f}'
                   .format(epoch, loss, current_loss, current_ch_loss, current_emd_loss))
        torch.save(model.cpu().state_dict(), model_path)  # saving model
        model.to(device)
        log_string('The model saved in {}'.format(model_path))
        best_loss = current_loss
        best_ch_loss = current_ch_loss
        best_emd_loss = current_emd_loss
        best_loss_epoch = epoch
        last_improvement = 0
    elif best_loss > 0:
        log_string(
            'Epoch {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Chamfer: {:.4f}, Test EMD: {:.4f},'
            ' Best Test Loss: {}, Best Test Chamfer: {:.4f}, Best Test EMD: {:.4f} Best Epoch: {}'.format(
                epoch, loss, current_loss, current_ch_loss, current_emd_loss, best_loss, best_ch_loss
                , best_emd_loss, best_loss_epoch
            ))
        last_improvement += 1
    else:
        log_string('Epoch {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Chamfer: {:.4f},'
                   ' Test EMD: {:.4f}'.format(epoch, loss, current_loss, current_ch_loss, current_emd_loss))
        last_improvement += 1

    if EARLY_STOPPING == 'True' and last_improvement > EARLY_STOPPING_PATIENCE:
        log_string('No improvement was observed after {} epochs.'.format(last_improvement))
        log_string(
            'The best model with the loss, chamfer, and emd of {}, {}, {}  on the validation set was saved at epoch {}.'
                .format(best_loss, best_ch_loss, best_emd_loss, best_loss_epoch
        ))
        break

if EARLY_STOPPING != 'True':
    log_string('The best model with the loss, emd, chamfer of {}, {}, {} on the validation set was saved at epoch {}.'
    .format(
        best_loss, best_ch_loss, best_emd_loss, best_loss_epoch
    ))