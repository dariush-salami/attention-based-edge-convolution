import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, BatchNorm
from custom_graph_convolution import CGCNConv
from temporal_edgecnn.temporal_edgecnn import TemporalSelfAttentionDynamicEdgeConv, TemporalDynamicEdgeConv, \
    AutomatedGraphDynamicEdgeConv, SelfTemporalSelfAttentionDynamicEdgeConv, TemporalDecoder
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).to(x.device)) \
            .view(1, 9).repeat(
            batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=4, aggr='max'):
        super().__init__()
        self.stn = STN3d()
        # self.stn2 = STN3d()
        # self.fstn = STNkd(k=64)
#         self.conv1 = AutomatedGraphDynamicEdgeConv(MLP([3, 16]),
#                                                    MLP([2 * 16, 64, 64, 64]),
#                                                    16, 64, 4, k, aggr)
#         self.conv2 = AutomatedGraphDynamicEdgeConv(None,
#                                                    MLP([2 * 64, 128]),
#                                                    64, 128, 8, k, aggr)

        self.conv1 = TemporalSelfAttentionDynamicEdgeConv(MLP([2 * 3, 64, 64, 64]),
                                                          64, 4, k, aggr)
        self.conv2 = TemporalSelfAttentionDynamicEdgeConv(MLP([2 * 64, 128]),
                                                          128, 8, k, aggr)

        self.deconv1 = TemporalSelfAttentionDynamicEdgeConv(MLP([2 * 3, 64, 64, 64]),
                                                          64, 4, k, aggr)
        self.deconv2 = TemporalSelfAttentionDynamicEdgeConv(MLP([2 * 64, 128]),
                                                          128, 8, k, aggr)
        self.deconv3 = TemporalDecoder(MLP([128, 128]), 128, 8, k, aggr)

        # self.conv1 = SelfTemporalSelfAttentionDynamicEdgeConv(MLP([2 * 3, 64, 64, 64]),
        #                                                   64, 4, k, aggr)
        # self.conv2 = SelfTemporalSelfAttentionDynamicEdgeConv(MLP([2 * 64, 128]),
        #                                                   128, 8, k, aggr)
        self.lin1 = MLP([128 + 64, 128])
        self.lin2 = MLP([128 + 64, 128])

        self.mlp = Seq(
            MLP([128, 256]), Dropout(0.5), MLP([256, 64]), Dropout(0.5),
            Lin(64, out_channels))

    def encode(self, data):
        sequence_numbers, pos, batch = data.x[:, 0].float(), data.x[:, 1:].float(), data.batch
        pos = pos.reshape(len(torch.unique(data.batch)), -1, 3).transpose(2, 1)
        trans = self.stn(pos)
        pos = pos.transpose(2, 1)
        pos = torch.bmm(pos, trans)
        pos = pos.reshape(-1, 3)
        x1 = self.conv1(pos, sequence_numbers, batch)
        x2 = self.conv2(x1, sequence_numbers, batch)
        # out = torch.cat([x1, x2], dim=1)
        return x1, x2, self.lin1(torch.cat([x1, x2], dim=-1))
        # return global_max_pool(out, batch)

    def decode(self, data, encoding):
        x1, x2, encoded = encoding
        sequence_number, pos, target, batch = data.x[:, 0].float(), data.x[:, 1:].float(), data.y[:, 1:].float(), data.batch
        starting_frame = pos[sequence_number == 10]
        target_without_end = target[sequence_number != 10]
        target = torch.cat([starting_frame.reshape([-1, 1, 128, 3]), target_without_end.reshape([-1, 9, 128, 3])],
                           dim=1).view(-1, 3)
        # trans = self.stn(target)
        # target = target.transpose(2, 1)
        # target = torch.bmm(target, trans)
        # target = target.reshape(-1, 3)
        dec1 = self.deconv1(target, sequence_number, batch)
        dec2 = self.deconv2(dec1, sequence_number, batch)
        out = self.lin2(torch.cat([dec1, dec2], dim=-1))

        out = self.deconv3(out, encoded, sequence_number, batch)
        return self.mlp(out)

    def frame_prod(self, decoding):
        return self.mlp(decoding)

    def forward(self, data):

        encoding = self.encode(data)

        decoding = self.decode(encoding)

        return self.frame_prod(decoding)
