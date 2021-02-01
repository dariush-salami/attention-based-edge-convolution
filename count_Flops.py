import torch
from torchvision.models import resnet50
from thop import profile, clever_format
model = resnet50()
input = torch.randn([1, 3, 224, 224])
macs, params = profile(model, inputs=(input, ))
from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)
