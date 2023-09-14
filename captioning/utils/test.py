# @Time : 2021/3/10 14:57 
# @Author : xx
# @File : test.py 
# @Software: PyCharm
import torch

a = torch.rand((3))

b = torch.zeros(3)
b[1] = 1
if b:
    a = a*2
else:
    a = a*10

print()