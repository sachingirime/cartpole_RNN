# Importing PyTorch and checking its version

import torch
torch.__version__

import numpy as np

a = [1,2,3]
a_ = np.array(a)
print(a_)

# Method-1
b = torch.from_numpy(a_)
print(b)

# Method-2
c = torch.tensor(a)
print(c.dtype)
print(c)

# Some must know parameters for tensor() function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d = torch.tensor([1,2,3], dtype = torch.float32, 
                 device = device, requires_grad = True)

print(d.shape)
print(d.dtype)
print(d.device)
print(d.requires_grad)

print(torch.zeros(3, 4, dtype=torch.float64))
print(torch.ones(4, 2, dtype=torch.float64))
print(torch.rand(3, 3, dtype=torch.float64))


a = torch.tensor([0, 1, 2, 3])

# boolean values
print(a.bool())

# Integer type values
print(a.short()) # int16
print(a.long()) # int64

# float type values
print(a.half()) # float16
print(a.double()) # float64


# Conversion from numpy array to tensor and vice-versa

import numpy as np

a = [1,2,3]
a_ = np.array(a)
print(a_)

# Numpy to Tensor
b = torch.from_numpy(a_)
print(b)

# Tensor to Numpy
c = b.numpy()
print(c)


a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([7, 8, 9], dtype=torch.float)

# Method-1
print(a + b)

# Method-2
print(torch.add(a, b))

# Method-3
c = torch.zeros(3)
c = torch.add(a, b, out=c)
print(c)

# Cumulative Sum
print(torch.add(a, b).sum())

a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([7, 8, 9], dtype=torch.float)

# Method-1
print(a - b)

# Method-2
print(torch.subtract(b, a))

# Method-3 (Variation)
c = torch.zeros(3)
c = torch.subtract(a, b, out=c)
print(c)

# Cumulative Sum of differences
print(torch.subtract(a, b).sum())

#Absolute cumulative Sum of differences
print(torch.abs(torch.subtract(a, b).sum()))


a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([7, 8, 9], dtype=torch.float)

# Method-1
print(a * b)

# Method-2
print(a.mul(b))

# Calculating the dot product
print(a.dot(b))


# Matrix multiplication 
# a shape of (m * n) and (n * p) will return a shape of (m * p)

a = torch.tensor([[1, 4, 2],[1, 5, 5]], dtype=torch.float)
b = torch.tensor([[5, 7],[8, 6],[9, 11]], dtype=torch.float)

# 3 ways of performing matrix multiplication

print("Method-1: \n", torch.matmul(a, b))
print("\nMethod-2: \n", torch.mm(a, b))
print("\nMethod-3: \n", a@b)


a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([7, 8, 9], dtype=torch.float)
#division
# Method-1
print(a / b)

# Method-2
c = torch.true_divide(a, b)
print(c)

# Variation
c = torch.true_divide(b, a)
print(c)

#indexing and slicing

a = torch.tensor(np.arange(0,10).reshape(2,5))
print(a)

# Indexing of tensors
print(a[0])
print(a[0][0])

# Tensor slicing
print(a[:, 0:3])

