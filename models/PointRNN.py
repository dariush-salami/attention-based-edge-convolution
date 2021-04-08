
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from modules.pointrnn_cell_impl import PointRNNCell


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class PointRnn(torch.nn.Module):
    
    
    def __init__(self, seq_length, num_points=128, num_samples=4, knn=False):
        
        super().__init__()
        self.seq_length = seq_length
        self.cell1 = PointRNNCell(radius=4.0+1e-6, nsamples=2*num_samples, in_channels = 0, out_channels=64)
        self.cell2 = PointRNNCell(radius=8.0+1e-6, nsamples=2*num_samples ,in_channels = 64, out_channels=128)
        self.cell3 = PointRNNCell(radius=12.0+1e-6, nsamples=2*num_samples, in_channels = 128, out_channels=256)
    
        self.mlp = MLP([256, 64])
        
        self.lin = Lin(64, 3)
    
    
    def encode(self, frames):
        
        states1 = None
        states2 = None
        states3 = None
        
        for i in range(int(self.seq_length/2)):
            states1 = self.cell1((frames[:, i].contiguous(), None), states1)
            
            states2 = self.cell2(states1, states2)
            
            states3 = self.cell3(states2, states3)
        
        return states1, states2, states3
        
            
    def decode(self, frames, encoded):
        
        states1, states2, states3 = encoded
            
        predicted_motions = []
        predicted_frames = []    
        
        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = frames[:, int(self.seq_length/2)-1].contiguous()
        for i in range(int(self.seq_length/2), self.seq_length):
            
            states1 = self.cell1((input_frame, None), states1)
            
            states2 = self.cell2(states1, states2)
            
            states3 = self.cell3(states2, states3)
            
            s_xyz3, s_feat3 = states3
            
            
            s_feat3_trans = torch.transpose(s_feat3, -1, 1)
            
            
            predicted_motion = self.mlp(s_feat3_trans)
            
            predicted_motion = self.lin(predicted_motion)

            predicted_motions.append(predicted_motion)
            input_frame += predicted_motion
            predicted_frames.append(input_frame)
            
        return predicted_frames
            
    def forward(self, frames):
        
        encoded = self.encode(frames)
        decoded = self.decode(frames, encoded)
        
        return decoded