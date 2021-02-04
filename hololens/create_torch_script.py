import torch
from hololens.model import Net
import torch_sparse
import torch_scatter
from pathlib import Path

print(torch_sparse.__version__)
print(torch_scatter.__version__)
model_path = '/home/researcher/PycharmProjects/RI4DPC/hololens/model.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net(21, k=2, T=1000, spatio_temporal_factor=10).to(device)
checkpoint = torch.load(Path(model_path))
model.load_state_dict(checkpoint)
model = torch.jit.script(model)
model.save("torch_script_model.pt")
print('The checkpoint was load successfully: {}'.format(model_path))

