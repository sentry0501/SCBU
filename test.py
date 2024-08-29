import torch
import numpy as np
model = torch.load("best.pt")
print(model)
# input = torch.randn((1,2,3,480,640))
batch_size =1
input = torch.randn(batch_size,3,480,640, requires_grad=False)
torch_out = model(input)
print("out pytorch",torch_out)
print("out pytorch",torch_out.dtype,torch_out.shape)

