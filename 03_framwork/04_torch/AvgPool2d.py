import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

input = Variable(
    torch.Tensor([[[1, 3, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]], [[1, 3, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]]))
print("input shape", input.shape)
c = F.avg_pool1d(input, kernel_size=3, stride=2)
print(c)
print("c shape:", c.shape)

# m = nn.AvgPool2d(3, stride=2)
m = nn.AvgPool2d((2, 2), stride=(2, 2))
input = Variable(torch.randn(20, 18, 50, 32))  # bach是20,图片size是50*31,chanel是1８(通道是1８,也就是每张图有1８个fature map)
input = np.array([[[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                   [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]],
                  [[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                   [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]])  # size２*2*4*4
print("input shape:", input.shape)
input = Variable(torch.FloatTensor(input))
output = m(input)
print(output)
print("output shape:", output.shape)  # (2,2,2,2)
