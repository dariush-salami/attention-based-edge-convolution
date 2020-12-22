import os.path as osp
from os import makedirs
import importlib
import torch
from pantomime_dataset import PantomimeDataset
from temporal_transformer import TemporalTransformer
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
import sys
BASE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(osp.join(ROOT_DIR, 'models'))


parser = argparse.ArgumentParser(description='Configurations')
parser.add_argument('--model', type=str, default='modified_edgecnn',
                    help='Model to run on the data (stgcnn, dgcnn, tgcnn, modified_edgecnn) [default: tgcnn]')
parser.add_argument('--log_dir', default='stgcnn', help='Log dir [default: stgcnn]')
parser.add_argument('--k', default=5, help='Number of nearest points [default: 5]')
parser.add_argument('--t', default=2, help='Number of future frames to look at [default: 5]')
parser.add_argument('--gpu_id', default=0, help='GPU ID [default: 0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size [default: 32]')
FLAGS = parser.parse_args()
LOG_DIR = FLAGS.log_dir
K = FLAGS.k
T = FLAGS.t
GPU_ID = FLAGS.gpu_id
MODEL = importlib.import_module(FLAGS.model)
BATCH_SIZE = FLAGS.batch_size

NUM_CLASSES = 21


if not osp.exists(LOG_DIR):
    print('Creating the model checkpoint directory at {}'.format(LOG_DIR))
    makedirs(LOG_DIR)


model_path = osp.join(LOG_DIR, 'model.pth')

device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/pantomime')
transform = TemporalTransformer(k=K, t=T)
train_dataset = PantomimeDataset(path, True, pre_transform=transform)
test_dataset = PantomimeDataset(path, False, pre_transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

model = MODEL.Net(21).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.squeeze())
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


interval_to_save = 5
print('Selected model: {}'.format(MODEL))
for epoch in range(1, 5000):
    loss = train()
    if epoch % interval_to_save == 0:
        test_acc = test(test_loader)
        print('Epoch {:03d}, Train Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch, loss, test_acc))
        torch.save(model.cpu().state_dict(), model_path)  # saving model
        model.cuda()
        print('The model saved in {}'.format(model_path))
    else:
        print('Epoch {:03d}, Train Loss: {:.4f}'.format(epoch, loss))
